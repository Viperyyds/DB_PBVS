"""
ModbusInterface_v2

提供一个基于单线程 ThreadPoolExecutor 的同步-to-async 包装器，
支持两种后端：
- 'modbuscommunicator'：使用项目内的 Communication.ModBusCommunicator
- 'pymodbus'：直接用 pymodbus.client.sync.ModbusTcpClient（如果安装了 pymodbus）

目标：兼容现有的 ModBusService 使用方式（enqueue -> _run_sync）

注意：本文件不再直接使用低级的线程锁；所有对底层 client 的访问都在单独的 executor 线程中串行执行。
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any
import logging

from Communication.CompactEntry import AddressBook, CompactEntry
from Communication.ModBusCommunicator import ModBusCommunicator

try:
    from pymodbus.client.sync import ModbusTcpClient as PyModbusTcpClient  
except Exception:
    PyModbusTcpClient = None  
    _HAS_PYMODBUS = False

LOGGER = logging.getLogger(__name__)


class ModbusInterfaceV2:
    """同步 Modbus 客户端的异步包装器，在单线程执行器中执行

    主要功能：通过 ThreadPoolExecutor 将同步操作转换为异步接口（_run_in_executor），确保不阻塞事件循环

    通讯选项[backend]: 'modbuscommunicator' or 'pymodbus'
    """
    # 一、 初始化
    def __init__(self, host: str, port: int, backend: str = 'modbuscommunicator', unit_id: int = 1,
                 address_book: Optional[Dict[str, CompactEntry]] = None, swap_words: bool = False,
                 swap_bytes_in_word: bool = False):
        self.host = host
        self.port = port
        self.unit_id = unit_id
        self.address_book = address_book
        self.backend = backend.lower()

        # 1. 创建executor 单线程保证对底层 client 的串行访问
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._loop = asyncio.get_event_loop()

        # 2. 两种后端对象选取
        self._com: Optional[ModBusCommunicator] = None
        self._pymod_client: Optional[Any] = None

        # 初始化后端
        if self.backend == 'modbuscommunicator':
            # 使用自写的 ModBusCommunicator 包装器
            self._com = ModBusCommunicator(self.host, self.port, unit_id=self.unit_id)
        elif self.backend == 'pymodbus':
            # 使用python自带的pymodbus
            if not _HAS_PYMODBUS or PyModbusTcpClient is None:
                raise RuntimeError('pymodbus not available')
            # 创建 pymodbus client
            self._pymod_client = PyModbusTcpClient(host=self.host, port=self.port)
        else:
            raise ValueError('Unsupported backend: %s' % backend)

    # 二、连接/断开连接
    async def connect(self) -> bool:
        return await self._run_in_executor(self._connect_sync)

    def _connect_sync(self) -> bool:
        try:
            if self.backend == 'modbuscommunicator' and self._com is not None:
                return self._com.connect()
            elif self.backend == 'pymodbus' and self._pymod_client is not None:
                return self._pymod_client.connect()
        except Exception:
            LOGGER.exception('connect failed')
        return False

    async def disconnect(self) -> bool:
        return await self._run_in_executor(self._disconnect_sync)

    def _disconnect_sync(self) -> bool:
        try:
            if self.backend == 'modbuscommunicator' and self._com is not None:
                return self._com.disconnect()
            elif self.backend == 'pymodbus' and self._pymod_client is not None:
                return self._pymod_client.close()
        except Exception:
            LOGGER.exception('disconnect failed')
        return False

    # helper : 在单线程执行器中运行同步 fn
    async def _run_in_executor(self, fn):
        return await self._loop.run_in_executor(self._executor, fn)


    # 三、通用包装器
    # ModBusService 使用的通用包装器
    # 这些方法反映了 ModBusService 中使用的 ModBusCommunicator API

    def _ensure_book(self):
        if self.address_book is None:
            raise RuntimeError('Address book not loaded')
        return self.address_book

    # read/write bool
    def _read_bool_sync(self, key: str, *idx: int) -> bool:
        self._ensure_book()
        book = self._ensure_book()
        if self._com is not None:
            return self._com.read_bool(book, key, *idx)
        # pymodbus raw path: resolve address first
        area, addr, words = AddressBook.address_of(book, key, *idx)
        client = self._pymod_client
        if client is None:
            raise RuntimeError('pymodbus client not initialized')
        if area.lower() == ModBusCommunicator.AREA_COILS.lower():
            rr = client.read_coils(addr, 1, unit=self.unit_id)
            if rr is None:
                raise RuntimeError(f"read_coils returned None for addr={addr}")
            bits = getattr(rr, 'bits', None)
            if not bits:
                raise RuntimeError(f"read_coils returned empty bits for addr={addr}")
            return bool(bits[0])
        if area.lower() == ModBusCommunicator.AREA_INPUTSTATUS.lower():
            rr = client.read_discrete_inputs(addr, 1, unit=self.unit_id)
            if rr is None:
                raise RuntimeError(f"read_discrete_inputs returned None for addr={addr}")
            bits = getattr(rr, 'bits', None)
            if not bits:
                raise RuntimeError(f"read_discrete_inputs returned empty bits for addr={addr}")
            return bool(bits[0])
        raise ValueError('Unsupported bit area: %s' % area)

    def _write_bool_sync(self, key: str, value: bool, *idx: int) -> None:
        book = self._ensure_book()
        if self._com is not None:
            return self._com.write_bool(book, key, value, *idx)
        area, addr, words = AddressBook.address_of(book, key, *idx)
        client = self._pymod_client
        if client is None:
            raise RuntimeError('pymodbus client not initialized')
        if area.lower() != ModBusCommunicator.AREA_COILS.lower():
            raise ValueError('Only Coils writable')
        client.write_coil(addr, value, unit=self.unit_id)

    # read/write uint16
    def _read_uint_sync(self, key: str, *idx: int) -> int:
        book = self._ensure_book()
        if self._com is not None:
            return self._com.read_uint(book, key, *idx)
        area, addr, words = AddressBook.address_of(book, key, *idx)
        client = self._pymod_client
        if client is None:
            raise RuntimeError('pymodbus client not initialized')
        if area.lower() == ModBusCommunicator.AREA_INPUT.lower():
            rr = client.read_input_registers(addr, 1, unit=self.unit_id)
            if rr is None:
                raise RuntimeError(f'read_input_registers returned None for addr={addr}')
            regs = list(getattr(rr, 'registers', []))
        else:
            rr = client.read_holding_registers(addr, 1, unit=self.unit_id)
            if rr is None:
                raise RuntimeError(f'read_holding_registers returned None for addr={addr}')
            regs = list(getattr(rr, 'registers', []))
        if not regs:
            raise RuntimeError(f'registers empty for addr={addr}')
        return regs[0]

    def _write_uint_sync(self, key: str, value: int, *idx: int) -> None:
        book = self._ensure_book()
        if self._com is not None:
            return self._com.write_uint(book, key, value, *idx)
        area, addr, words = AddressBook.address_of(book, key, *idx)
        client = self._pymod_client
        if client is None:
            raise RuntimeError('pymodbus client not initialized')
        # allow both holding and input for backward compatibility
        if area.lower() == ModBusCommunicator.AREA_HOLDING.lower():
            client.write_register(addr, value, unit=self.unit_id)
        else:
            raise ValueError('Write uint only supported to HoldingRegisters')

    # read/write real (32-bit float)
    def _read_real_sync(self, key: str, *idx: int) -> float:
        book = self._ensure_book()
        if self._com is not None:
            return self._com.read_real(book, key, *idx)
        area, addr, words = AddressBook.address_of(book, key, *idx)
        client = self._pymod_client
        if client is None:
            raise RuntimeError('pymodbus client not initialized')
        if area.lower() == ModBusCommunicator.AREA_INPUT.lower():
            rr = client.read_input_registers(addr, 2, unit=self.unit_id)
        else:
            rr = client.read_holding_registers(addr, 2, unit=self.unit_id)
        if rr is None:
            raise RuntimeError(f'read registers returned None for addr={addr}')
        regs = list(getattr(rr, 'registers', []))
        if len(regs) < 2:
            raise RuntimeError(f'Not enough registers for float at addr={addr}')
        from Communication.ModBusUtils import registers_to_float
        return registers_to_float(regs[0], regs[1])

    def _write_real_sync(self, key: str, value: float, *idx: int) -> None:
        book = self._ensure_book()
        if self._com is not None:
            return self._com.write_real(book, key, value, *idx)
        area, addr, words = AddressBook.address_of(book, key, *idx)
        if area.lower() != ModBusCommunicator.AREA_HOLDING.lower():
            raise ValueError('Write real only supported to HoldingRegisters')
        client = self._pymod_client
        if client is None:
            raise RuntimeError('pymodbus client not initialized')
        from Communication.ModBusUtils import float_to_registers
        hi, lo = float_to_registers(value)
        client.write_registers(addr, [hi, lo], unit=self.unit_id)

    # read/write dword (32-bit unsigned int)
    def _read_dword_sync(self, key: str, *idx: int) -> int:
        book = self._ensure_book()
        if self._com is not None:
            return self._com.read_dword(book, key, *idx)
        area, addr, words = AddressBook.address_of(book, key, *idx)
        client = self._pymod_client
        if client is None:
            raise RuntimeError('pymodbus client not initialized')
        if area.lower() == ModBusCommunicator.AREA_INPUT.lower():
            rr = client.read_input_registers(addr, 2, unit=self.unit_id)
        else:
            rr = client.read_holding_registers(addr, 2, unit=self.unit_id)
        if rr is None:
            raise RuntimeError(f'read registers returned None for addr={addr}')
        regs = list(getattr(rr, 'registers', []))
        if len(regs) < 2:
            raise RuntimeError(f'Not enough registers for dword at addr={addr}')
        from Communication.ModBusUtils import registers_to_dword
        return registers_to_dword(regs[0], regs[1])

    def _write_dword_sync(self, key: str, value: int, *idx: int) -> None:
        book = self._ensure_book()
        if self._com is not None:
            return self._com.write_dword(book, key, value, *idx)
        area, addr, words = AddressBook.address_of(book, key, *idx)
        if area.lower() != ModBusCommunicator.AREA_HOLDING.lower():
            raise ValueError('Write dword only supported to HoldingRegisters')
        client = self._pymod_client
        if client is None:
            raise RuntimeError('pymodbus client not initialized')
        from Communication.ModBusUtils import dword_to_registers
        hi, lo = dword_to_registers(value)
        client.write_registers(addr, [hi, lo], unit=self.unit_id)

    # block read helpers used by ModBusService
    def _read_discrete_inputs_block_sync(self, start: int, count: int) -> bytes:
        if self._com is not None:
            return self._com.read_discrete_inputs_block(start, count)
        # pymodbus client: read discrete inputs in chunks
        client = self._pymod_client
        if client is None:
            raise RuntimeError('pymodbus client not initialized')
        result = bytearray((count + 7) // 8)
        remaining = count
        addr = start
        dst = 0
        while remaining > 0:
            to_read = min(remaining, 2000)
            rr = client.read_discrete_inputs(addr, to_read, unit=self.unit_id)
            if rr is None:
                raise RuntimeError(f'read_discrete_inputs returned None for addr={addr}')
            chunk = list(getattr(rr, 'bits', []))
            chunk_bytes = bytearray((to_read + 7) // 8)
            for i, bit in enumerate(chunk):
                if bit:
                    byte_index = i // 8
                    bit_index = i % 8
                    chunk_bytes[byte_index] |= (1 << bit_index)
            copy = min(len(chunk_bytes), len(result) - dst)
            result[dst:dst+copy] = chunk_bytes[:copy]
            addr += to_read
            remaining -= to_read
            dst += copy
        return bytes(result)

    def _read_input_registers_block_sync(self, start: int, count: int) -> list:
        if self._com is not None:
            return self._com.read_input_registers_block(start, count)
        client = self._pymod_client
        if client is None:
            raise RuntimeError('pymodbus client not initialized')
        result = []
        remaining = count
        addr = start
        while remaining > 0:
            to_read = min(remaining, 120)
            rr = client.read_input_registers(addr, to_read, unit=self.unit_id)
            if rr is None:
                raise RuntimeError(f'read_input_registers returned None for addr={addr}')
            result.extend(list(getattr(rr, 'registers', [])))
            addr += to_read
            remaining -= to_read
        return result

    # Async public APIs used by ModBusService
    async def write_bool(self, key: str, value: bool, *idx: int):
        return await self._run_in_executor(lambda: self._write_bool_sync(key, value, *idx))

    async def write_uint(self, key: str, value: int, *idx: int):
        return await self._run_in_executor(lambda: self._write_uint_sync(key, value, *idx))

    async def write_real(self, key: str, value: float, *idx: int):
        return await self._run_in_executor(lambda: self._write_real_sync(key, value, *idx))

    async def read_bool(self, key: str, *idx: int) -> bool:
        return await self._run_in_executor(lambda: self._read_bool_sync(key, *idx))

    async def read_uint(self, key: str, *idx: int) -> int:
        return await self._run_in_executor(lambda: self._read_uint_sync(key, *idx))

    async def read_real(self, key: str, *idx: int) -> float:
        return await self._run_in_executor(lambda: self._read_real_sync(key, *idx))

    async def write_dword(self, key: str, value: int, *idx: int):
        return await self._run_in_executor(lambda: self._write_dword_sync(key, value, *idx))

    async def read_dword(self, key: str, *idx: int) -> int:
        return await self._run_in_executor(lambda: self._read_dword_sync(key, *idx))

    async def read_discrete_inputs_block(self, start: int, count: int) -> bytes:
        return await self._run_in_executor(lambda: self._read_discrete_inputs_block_sync(start, count))

    async def read_input_registers_block(self, start: int, count: int) -> list:
        return await self._run_in_executor(lambda: self._read_input_registers_block_sync(start, count))


    def close(self):
        try:
            if self._pymod_client is not None:
                self._pymod_client.close()
            if self._com is not None:
                self._com.dispose()
        finally:
            try:
                self._executor.shutdown(wait=True)
            except Exception:
                pass

    # --- Test utilities (原 TestRead / TestWrite) ---
    def _test_read_sync(self, address: int) -> int:
        """读取单个保持寄存器的值（同步）。"""
        if self._com is not None:
            try:
                regs = self._com._read_registers(ModBusCommunicator.AREA_HOLDING, address, 1)
                if not regs:
                    return 0
                return regs[0]
            except Exception:
                LOGGER.exception('TestRead via ModBusCommunicator failed')
                return 0
        # pymodbus path
        client = self._pymod_client
        if client is None:
            raise RuntimeError('pymodbus client not initialized')
        rr = client.read_holding_registers(address, 1, unit=self.unit_id)
        if rr is None:
            return 0
        regs = list(getattr(rr, 'registers', []))
        if not regs:
            return 0
        return regs[0]

    def _test_write_sync(self, address: int, value: bool) -> bool:
        """写入单个保持寄存器的值（同步）。"""
        if self._com is not None:
            try:
                if hasattr(self._com, '_write_single_coil'):
                    return self._com._write_single_coil(address, value) is None or True
                LOGGER.warning('ModBusCommunicator lacks _write_single_coil; TestWrite not supported')
                return False
            except Exception:
                LOGGER.exception('TestWrite via ModBusCommunicator failed')
                return False
        client = self._pymod_client
        if client is None:
            raise RuntimeError('pymodbus client not initialized')

        try:
            if hasattr(client, 'write_coil'):
                client.write_coil(address, value, unit=self.unit_id)
            elif hasattr(client, 'write_single_coil'):
                client.write_single_coil(address, value, unit=self.unit_id)
            else:
                # fallback: try write_register with 0/1
                if hasattr(client, 'write_register'):
                    client.write_register(address, int(bool(value)), unit=self.unit_id)
                else:
                    raise RuntimeError('No suitable write method on pymodbus client')
            return True
        except Exception:
            LOGGER.exception('TestWrite via pymodbus failed')
            return False

    async def test_read(self, address: int) -> int:
        return await self._run_in_executor(lambda: self._test_read_sync(address))

    async def test_write(self, address: int, value: bool) -> bool:
        return await self._run_in_executor(lambda: self._test_write_sync(address, value))
    

    # ---------- High-level setters using address-book keys (async) ----------
    async def set_joints_target_position(self, positions: list):
        """Set target positions for 6 joints. positions is iterable of 6 floats."""
        if len(positions) < 6:
            raise ValueError('positions must contain at least 6 elements')
        # keys must exist in the address book and correspond to these names
        await self.write_real('Joint1_Target_Position', float(positions[0]))
        await self.write_real('Joint2_Target_Position', float(positions[1]))
        await self.write_real('Joint3_Target_Position', float(positions[2]))
        await self.write_real('Joint4_Target_Position', float(positions[3]))
        await self.write_real('Joint5_Target_Position', float(positions[4]))
        await self.write_real('Joint6_Target_Position', float(positions[5]))

    async def set_cartesian_target_position(self, pose: list):
        """Set TCP target pose [x,y,z,roll,pitch,yaw]."""
        if len(pose) < 6:
            raise ValueError('pose must contain at least 6 elements')
        await self.write_real('TCP_Target_Position_X', float(pose[0]))
        await self.write_real('TCP_Target_Position_Y', float(pose[1]))
        await self.write_real('TCP_Target_Position_Z', float(pose[2]))
        await self.write_real('TCP_Target_Position_Roll', float(pose[3]))
        await self.write_real('TCP_Target_Position_Pitch', float(pose[4]))
        await self.write_real('TCP_Target_Position_Yaw', float(pose[5]))

    async def set_tcp_move_rel_position(self, positions: list):
        """Set TCP relative move distances/angles [dx,dy,dz,rx,ry,rz]."""
        if len(positions) < 6:
            raise ValueError('positions must contain at least 6 elements')
        await self.write_real('TCP_Relative_Distance_along_X', float(positions[0]))
        await self.write_real('TCP_Relative_Distance_along_Y', float(positions[1]))
        await self.write_real('TCP_Relative_Distance_along_Z', float(positions[2]))
        await self.write_real('TCP_Relative_Distance_around_X', float(positions[3]))
        await self.write_real('TCP_Relative_Distance_around_Y', float(positions[4]))
        await self.write_real('TCP_Relative_Distance_around_Z', float(positions[5]))

    async def set_joint_move_abs_position(self, joint_index: int, position: float):
        """Set a single joint's ABS target position by joint index (1-6).

        This is the v2 async equivalent of the original SetJointMoveABSPosition.
        It resolves the target register key via the address-book key names and
        uses the async write_real wrapper to perform the write.
        """
        keys = {
            1: 'Joint1_Target_Position',
            2: 'Joint2_Target_Position',
            3: 'Joint3_Target_Position',
            4: 'Joint4_Target_Position',
            5: 'Joint5_Target_Position',
            6: 'Joint6_Target_Position',
        }
        key = keys.get(int(joint_index))
        if key is None:
            raise ValueError(f'index error: invalid joint_index={joint_index}, expected 1..6')
        await self.write_real(key, float(position))

    # backward-compatible camelCase/legacy name for callers that expect it
    async def SetJointMoveABSPosition(self, joint_index: int, position: float):
        return await self.set_joint_move_abs_position(joint_index, position)

    


__all__ = ['ModbusInterfaceV2']
