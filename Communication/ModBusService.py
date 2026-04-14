"""异步 ModBusService
    Python 版本的 ModBusService

asyncio 实现：
- 单消费者消费者队列串行执行所有 Modbus I/O 操作
- 周期性轮询（默认 40ms），合并任务避免避免任务堆积
- 脉冲写（PulseBoolAsync）使用 asyncio.Lock 按 key 序列化操作
- 支持重连退避策略（失败次数越多，重连间隔越长）
"""

import asyncio
from typing import Dict, Callable, Optional, Any, List
from datetime import datetime, timedelta, timezone

from Communication.ModBusCommunicator import ModBusCommunicator
from Communication.CompactEntry import CompactEntry, AddressBook
from Communication.ModBusUtils import registers_to_float, registers_to_dword


class RobotStatus:
    """机器人状态信息类"""

    AXES = 6  # TODO 关节数量 (可根据实际情况修改)
    
    def __init__(self):
        n = self.AXES
        # 基础状态标志
        self.Initialized = False  # 初始化完成标志
        self.PowerOn = False      # 上电状态
        self.Moving = False       # 运动状态
        self.Error = False        # 错误状态
        self.ErrorId = 0          # 错误代码
        
        # 关节状态（数组长度与关节数一致）
        self.JointError = [False] * n       # 关节错误状态
        self.JointMoving = [False] * n      # 关节运动状态
        self.JointErrorId = [0] * n         # 关节错误代码
        self.JointState = [0] * n           # 关节状态码
        self.JointActualPosition = [0.0] * n  # 关节实际位置
        self.JointActualVelocity = [0.0] * n  # 关节实际速度
        self.JointActualAcceleration = [0.0] * n  # 关节实际加速度
        self.JointActualCurrent = [0.0] * n     # 关节实际电流
        self.JointActualTorque = [0.0] * n      # 关节实际扭矩
        
        # 法兰盘状态（6个元素：位置/姿态等）
        self.FlangePose = [0.0] * 6       # 法兰盘位姿
        self.FlangeTwist = [0.0] * 6      # 法兰盘速度
        self.FlangeWrench = [0.0] * 6     # 法兰盘力/力矩
        
        # TCP（工具中心点）状态
        self.TcpPose = [0.0] * 6          # TCP位姿
        self.TcpTwist = [0.0] * 6         # TCP速度
        self.TcpWrench = [0.0] * 6        # TCP力/力矩


class ModBusService:
    # 外部回调函数（同步或异步均可）：状态更新时触发
    snapshot_updated: Optional[Callable[[RobotStatus], Any]] = None

    # 轮询间隔默认为 40ms
    def __init__(self, communicator: ModBusCommunicator, address_book: Dict[str, CompactEntry], poll_interval: float = 0.04):
        self._com = communicator          # Modbus通信器实例
        self._book = address_book         # 地址簿（寄存器地址与功能的映射）
        self._poll_interval = poll_interval  # 轮询间隔（秒）

        self._pulse_locks: Dict[str, asyncio.Lock] = {}  # 脉冲操作的锁字典（按key区分）
        # 队列和任务占位符：队列在start()中创建，未启动前为None，目的是未启动服务时调用enqueue能抛出明确错误，而非阻塞
        self._queue: Optional[asyncio.Queue] = None
        self._consumer_task: Optional[asyncio.Task] = None  # 消费队列的任务
        self._poller_task: Optional[asyncio.Task] = None     # 轮询任务
        self._running = False                               # 服务运行状态

        # 事件循环占位符：在start()中设置
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # 合并投递标志：避免同一时间投递多个轮询任务
        self._poll_work_pending = False

        # 重连退避参数
        self._consecutive_failures = 0          # 连续失败次数
        self._reconnect_backoff = timedelta(milliseconds=500)      # 当前重连间隔
        self._reconnect_backoff_min = timedelta(milliseconds=500)  # 最小重连间隔
        self._reconnect_backoff_max = timedelta(seconds=5)         # 最大重连间隔
        
        # 使用UTC时间避免时区问题
        self._last_reconnect_attempt = datetime.min.replace(tzinfo=timezone.utc)

        # 外部回调函数（同步或异步均可）：状态更新时触发
        self.snapshot_updated: Optional[Callable[[RobotStatus], Any]] = None

    @property
    def is_running(self) -> bool:
        """判断服务是否正在运行"""
        return self._running and self._consumer_task is not None and not self._consumer_task.done()

    # 启动服务
    async def start(self):
        if self.is_running:
            return
        # 记录当前事件循环（服务运行的循环）并创建绑定到该循环的队列
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        self._running = True
        
        # 创建消费队列和轮询的任务
        self._consumer_task = asyncio.create_task(self._consume_loop())
        self._poller_task = asyncio.create_task(self._poll_loop())

        # 将连接操作加入队列
        await self.enqueue(lambda: self._run_sync(self._com.connect))
    
    # 停止服务
    async def stop(self):
        if not self.is_running:
            return
        self._running = False

        # 1. 取消轮询任务
        if self._poller_task:
            self._poller_task.cancel()
            try:
                await asyncio.wait_for(self._poller_task, timeout=2.0)
            except Exception:
                pass

        # 2. 向队列发送终止信号（让消费循环退出阻塞）
        if self._queue is not None:
            try:
                # 放入None作为终止信号（消费循环会处理）
                await self._queue.put(None)
            except Exception:
                print("向队列放入终止信号失败")

        # 3. 等待消费任务退出
        if self._consumer_task:
            try:
                await asyncio.wait_for(self._consumer_task, timeout=5.0)
            except Exception:
                pass
        
        # 4. 断开连接
        try:
            self._com.disconnect()
        except Exception:
            print("断开连接失败")
            
    async def dispose(self):
        """释放资源（本质是停止服务）"""
        await self.stop()

    # 公开的异步API：将操作加入队列
    async def write_bool(self, key: str, value: bool, *idx: int):
        """写入布尔值（线圈）"""
        return await self.enqueue(lambda: self._run_sync(lambda: self._com.write_bool(self._book, key, value, *idx)))

    async def write_uint(self, key: str, value: int, *idx: int):
        """写入16位无符号整数"""
        return await self.enqueue(lambda: self._run_sync(lambda: self._com.write_uint(self._book, key, value, *idx)))

    async def write_real(self, key: str, value: float, *idx: int):
        """写入浮点数（32位）"""
        return await self.enqueue(lambda: self._run_sync(lambda: self._com.write_real(self._book, key, value, *idx)))

    async def read_bool(self, key: str, *idx: int) -> bool:
        """读取布尔值（线圈或离散输入）"""
        return await self.enqueue(lambda: self._run_sync(lambda: self._com.read_bool(self._book, key, *idx)))

    async def read_uint(self, key: str, *idx: int) -> int:
        """读取16位无符号整数"""
        return await self.enqueue(lambda: self._run_sync(lambda: self._com.read_uint(self._book, key, *idx)))

    async def read_real(self, key: str, *idx: int) -> float:
        """读取浮点数（32位）"""
        return await self.enqueue(lambda: self._run_sync(lambda: self._com.read_real(self._book, key, *idx)))

    async def write_dword(self, key: str, value: int, *idx: int):
        """写入双字（32位整数）"""
        return await self.enqueue(lambda: self._run_sync(lambda: self._com.write_dword(self._book, key, value, *idx)))

    async def read_dword(self, key: str, *idx: int) -> int:
        """读取双字（32位整数）"""
        return await self.enqueue(lambda: self._run_sync(lambda: self._com.read_dword(self._book, key, *idx)))

    async def pulse_bool(self, key: str, milliseconds: int):
        """脉冲写入布尔值：先置为True，延迟指定毫秒后重置为False"""
        # 按key获取锁，确保同一key的脉冲操作串行执行
        lock = self._pulse_locks.setdefault(key, asyncio.Lock())
        async with lock:
            await self.write_bool(key, True)
            await asyncio.sleep(milliseconds / 1000.0)  # 转换为秒
            try:
                await self.write_bool(key, False)
            except Exception:
                print("脉冲重置失败")
                
    async def read_snapshot(self) -> Optional[RobotStatus]:
        """读取一次完整的机器人状态快照"""
        return await self.enqueue(lambda: self._run_sync(self._poll_once_safe))

    # 内部队列操作辅助方法
    async def enqueue(self, coro_factory):
        """将任务加入队列，确保所有操作串行执行"""
        if self._queue is None:
            raise RuntimeError("服务未启动")

        # 获取调用方的事件循环和创建未来对象（用于返回结果）
        caller_loop = asyncio.get_running_loop()
        caller_fut = caller_loop.create_future()

        # 包装任务：执行并处理结果/异常
        async def wrapper():
            try:
                res = coro_factory()
                # 如果是协程，等待其执行完成
                if asyncio.iscoroutine(res):
                    res = await res
                # 向调用方的未来对象设置结果（线程安全）
                caller_loop.call_soon_threadsafe(caller_fut.set_result, res)
            except Exception as ex:
                # 向调用方的未来对象设置异常（线程安全）
                caller_loop.call_soon_threadsafe(caller_fut.set_exception, ex)

        # 如果调用方与服务在同一个事件循环，直接放入队列
        if hasattr(self, '_loop') and self._loop is caller_loop:
            await self._queue.put(wrapper)
        else:
            # 跨事件循环调用，使用线程安全的方式加入队列
            if self._loop is None:
                raise RuntimeError("服务事件循环未设置")
            # print("跨事件循环，通过线程安全方式加入队列")
            asyncio.run_coroutine_threadsafe(self._queue.put(wrapper), self._loop)

        # 等待任务执行完成并返回结果
        return await caller_fut

    async def _consume_loop(self):
        """消费队列的循环：串行执行所有Modbus操作"""
        while self._running:
            try:
                # 确保队列已初始化
                queue = self._queue
                if queue is None:
                    raise RuntimeError("服务队列未初始化")
                # 从队列获取任务（阻塞直到有任务）
                work = await queue.get()
                # 处理终止信号：如果是None，退出循环
                if work is None:
                    print("收到终止信号，退出消费循环")
                    break

                await work()  # 执行任务

                # 任务成功执行，重置失败计数和退避间隔
                self._consecutive_failures = 0
                self._reconnect_backoff = self._reconnect_backoff_min
            except asyncio.CancelledError:
                break  # 任务被取消，退出循环

            except Exception as ex:
                # 任务执行失败，更新失败计数
                self._consecutive_failures += 1
                print(f"I/O操作失败（第{self._consecutive_failures}次）：{ex}")
                # 重连逻辑
                now = datetime.now(timezone.utc)
                
                # 连续失败3次以上，且距离上次重连尝试已超过退避间隔
                threshold_reached = self._consecutive_failures >= 3
                cooldown_passed = (now - self._last_reconnect_attempt) >= self._reconnect_backoff
                if threshold_reached and cooldown_passed:
                    print(f"连续失败{self._consecutive_failures}次，尝试重连；退避间隔={self._reconnect_backoff}")
                    try:
                        self._com.disconnect()  # 先断开旧连接
                    except Exception:
                        pass
                    try:
                        self._com.connect()  # 尝试重新连接
                    except Exception:
                        print("重连失败")
                    self._last_reconnect_attempt = now
                    # 指数退避：下次间隔翻倍（不超过最大值）
                    self._reconnect_backoff = min(self._reconnect_backoff * 2, self._reconnect_backoff_max)
                await asyncio.sleep(0.1)  # 失败后短暂等待

    async def _poll_loop(self):
        """周期性轮询循环：按间隔向队列投递状态读取任务"""
        print(f"启动轮询循环，间隔{self._poll_interval}秒")
        try:
            while self._running:
                await asyncio.sleep(self._poll_interval)  # 等待轮询间隔
                # 如果已有轮询任务在队列中，跳过本次（避免堆积）
                if self._poll_work_pending:
                    continue
                self._poll_work_pending = True  # 标记有未完成的轮询任务
                # 定义轮询任务
                async def poll_work():
                    try:
                        # 安全地读取一次状态快照
                        snap = self._poll_once_safe()
                        # 如果有外部回调，触发回调（支持异步回调）
                        if snap is not None and self.snapshot_updated:
                            try:
                                res = self.snapshot_updated(snap)
                                if asyncio.iscoroutine(res):
                                    await res  # 异步回调需要等待完成
                            except Exception:
                                print("snapshot_updated回调执行失败")
                    finally:
                        self._poll_work_pending = False  # 重置标记

                # 将轮询任务加入队列
                if self._queue is None:
                    raise RuntimeError("服务队列未初始化")
                await self._queue.put(poll_work)
        except asyncio.CancelledError:
            print("轮询循环被取消")

    def _run_sync(self, fn):
        """执行同步函数并返回结果。用于在队列包装器中调用同步方法"""
        return fn()

    def _poll_once_safe(self) -> Optional[RobotStatus]:
        """安全地读取一次完整的机器人状态（包含异常捕获）"""
        try:
            # 辅助函数：获取地址簿中指定key的寄存器地址
            def Addr(key: str) -> int:
                t = AddressBook.address_of(self._book, key)
                return t[1]  # t是(区域, 地址, 长度)，取地址部分

            # 辅助函数：获取带索引的key的寄存器地址（如数组元素）
            def AddrI(key: str, i: int) -> int:
                t = AddressBook.address_of(self._book, key, i + 1)  # i+1是因为地址簿可能从1开始索引
                return t[1]

            # 定义需要读取的寄存器范围
            FlagsStart = Addr("Flags.Initialized")  # 标志位起始地址
            FlagsCount = 128                        # 标志位数量
            StatusStart = Addr("Status.Error_ID")   # 状态寄存器起始地址
            StatusCount = 158                       # 状态寄存器数量
            
            # 批量读取离散输入（标志位）和输入寄存器（状态数据）
            byte_flags = self._com.read_discrete_inputs_block(FlagsStart, FlagsCount)
            regs_status = self._com.read_input_registers_block(StatusStart, StatusCount)
            
            # 解析数据到RobotStatus对象
            s = RobotStatus()
            
            # 基础状态标志（从离散输入中解析）
            s.Initialized = self._get_bit(byte_flags, FlagsStart, Addr("Flags.Initialized"))
            s.PowerOn = self._get_bit(byte_flags, FlagsStart, Addr("Flags.PowerOn"))
            s.Moving = self._get_bit(byte_flags, FlagsStart, Addr("Flags.Moving"))
            s.Error = self._get_bit(byte_flags, FlagsStart, Addr("Flags.Error"))

            # 错误代码（从状态寄存器中解析）
            s.ErrorId = self._get_dword(regs_status, StatusStart, Addr("Status.Error_ID"))

            # 关节状态（数组）
            s.JointError = [self._get_bit(byte_flags, FlagsStart, AddrI("Flags.Joint_Error", i)) for i in range(6)]
            s.JointMoving = [self._get_bit(byte_flags, FlagsStart, AddrI("Flags.Joint_Moving", i)) for i in range(6)]

            s.JointErrorId = [self._get_dword(regs_status, StatusStart, AddrI("Status.Joint_Error_ID", i)) for i in range(6)]
            s.JointState = [self._get_dword(regs_status, StatusStart, AddrI("Status.Joint_State", i)) for i in range(6)]

            # 关节运动数据（浮点数）
            s.JointActualPosition = [float(self._get_f32(regs_status, StatusStart, AddrI("Status.Joint_Actual_Position", i))) for i in range(6)]
            s.JointActualVelocity = [float(self._get_f32(regs_status, StatusStart, AddrI("Status.Joint_Actual_Velocity", i))) for i in range(6)]
            s.JointActualAcceleration = [float(self._get_f32(regs_status, StatusStart, AddrI("Status.Joint_Actual_Acceleration", i))) for i in range(6)]
            s.JointActualCurrent = [float(self._get_f32(regs_status, StatusStart, AddrI("Status.Joint_Actual_Current", i))) for i in range(6)]
            s.JointActualTorque = [float(self._get_f32(regs_status, StatusStart, AddrI("Status.Joint_Actual_Torque", i))) for i in range(6)]

            # 法兰盘和TCP数据（浮点数）
            s.FlangePose = [float(self._get_f32(regs_status, StatusStart, AddrI("Status.Flange_Pose", i))) for i in range(6)]
            s.FlangeTwist = [float(self._get_f32(regs_status, StatusStart, AddrI("Status.Flange_Twist", i))) for i in range(6)]
            s.FlangeWrench = [float(self._get_f32(regs_status, StatusStart, AddrI("Status.Flange_Wrench", i))) for i in range(6)]
            s.TcpPose = [float(self._get_f32(regs_status, StatusStart, AddrI("Status.TCP_Pose", i))) for i in range(6)]
            s.TcpTwist = [float(self._get_f32(regs_status, StatusStart, AddrI("Status.TCP_Twist", i))) for i in range(6)]
            s.TcpWrench = [float(self._get_f32(regs_status, StatusStart, AddrI("Status.TCP_Wrench", i))) for i in range(6)]
            
            # 打印出读取的状态以供调试
            # print("轮询状态成功：", s.__dict__)

            return s
        except Exception:
            print("轮询状态失败，跳过本次循环")
            return None

    # 数据解析辅助方法
    @staticmethod
    def _get_bit(packed_bits: bytes, block_start: int, absolute_bit_address: int) -> bool:
        """从打包的字节数组中解析指定地址的位状态"""
        bit_offset = absolute_bit_address - block_start  # 计算相对偏移
        if bit_offset < 0:
            return False
        byte_index = bit_offset // 8  # 计算所在字节索引
        bit_in_byte = bit_offset % 8  # 计算字节内的位索引
        # 检查索引是否有效
        if byte_index < 0 or byte_index >= len(packed_bits):
            return False
        # 位运算判断该位是否为1
        return (packed_bits[byte_index] & (1 << bit_in_byte)) != 0

    @staticmethod
    def _get_uint16(regs: List[int], block_start: int, absolute_reg_address: int) -> int:
        """从寄存器列表中解析16位无符号整数"""
        reg_offset = absolute_reg_address - block_start  # 相对偏移
        if reg_offset < 0 or reg_offset >= len(regs):
            return 0
        return regs[reg_offset]

    def _get_dword(self, regs: List[int], block_start: int, absolute_reg_address: int) -> int:
        """从寄存器列表中解析32位整数（双字）"""
        reg_offset = absolute_reg_address - block_start
        # 确保有连续两个寄存器可用
        if reg_offset < 0 or reg_offset + 1 >= len(regs):
            return 0
        reg0 = regs[reg_offset]
        reg1 = regs[reg_offset + 1]
        return registers_to_dword(reg0, reg1)  # 调用工具函数转换

    def _get_f32(self, regs: List[int], block_start: int, absolute_reg_address: int) -> float:
        """从寄存器列表中解析32位浮点数"""
        reg_offset = absolute_reg_address - block_start
        # 确保有连续两个寄存器可用（浮点数占2个16位寄存器）
        if reg_offset < 0 or reg_offset + 1 >= len(regs):
            return 0.0
        hi = regs[reg_offset]    # 高位寄存器
        lo = regs[reg_offset + 1]  # 低位寄存器
        return registers_to_float(hi, lo)  # 调用工具函数转换


__all__ = ['ModBusService', 'RobotStatus']