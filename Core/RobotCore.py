# RobotCore.py 开头
import sys
import os

import time
from time import sleep
import threading

import asyncio
from typing import Optional, List, Union
from threading import Timer

from .Basic import *
from Communication.ModBusService import ModBusService, RobotStatus
from Communication.ModBusCommunicator import ModBusCommunicator
from Communication.CompactEntry import AddressBook

import json
from pprint import pprint
from .Util import *

class RobotCore:
    def __init__(self,target_ip:str):
        # 创建底层同步 communicator
        # TODO 在设备上验证需要开启 swap_words=True
        self.communicator = ModBusCommunicator(target_ip, 502, unit_id=1, swap_words=True)

        # 加载地址簿
        self._address_book = self._load_address_book()

        # TODO 加载默认参数json文件地址，按需修改json文件路径，根目录为 SDKPython
        self.default_parameter_json = "./Communication/DefaultRobotParameters.json"

        # 创建 ModBusService（async 管理 I/O 队列）
        self._service = ModBusService(self.communicator, self._address_book or {})

        # 事件循环与后台线程
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread = None

        # 机器人状态与参数
        self.robot_parameter = RobotParameters(json_file = self.default_parameter_json)
        self.robot_status = RobotStatus()
        self.timer = Timer(0.05, self.ReadRobotStatus)
        # 确保定时器线程为守护线程，以免阻止进程退出
        self.timer.daemon = True

        # 连接状态标记
        self.connected = False

        # 尝试连接并初始化
        self._initialize_connection()

    def _load_address_book(self) -> Optional[dict]:
        """加载地址簿的工具方法"""
        # TODO 按需修改json文件路径，根目录为 SDKPython
        try_paths = ['./Communication/modbus_address_book.compact_win.json']
        for path in try_paths:
            try:
                return AddressBook.load(path)
            except Exception as e:
                print(f"加载地址簿失败（路径：{path}）：{e}")
        return None

    def _initialize_connection(self) -> None:
        """初始化连接并启动服务"""
        try:
            self.connected = self.communicator.connect()
        except Exception as e:
            print(f"连接机器人失败：{e}")
            self.connected = False

        if self.connected:
            print("连接成功！启动 ModBusService...")
            self._start_service_loop()
            self.timer.start()
            self._initialize_robot()
        else:
            print("连接失败！")

    def _start_service_loop(self) -> None:
        """启动事件循环线程"""
        self._loop = asyncio.new_event_loop()

        # 启动 ModBusService
        loop = self._loop
        assert loop is not None
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever,
            daemon=True
        )
        self._loop_thread.start()

        # 在新启动的事件循环中启动 ModBusService 的异步任务（consumer/poller）
        # 这会在后台事件循环中调用 ModBusService.start()，确保队列被创建并开始消费
        try:
            fut = asyncio.run_coroutine_threadsafe(self._service.start(), self._loop)
            # 短暂等待服务启动，避免在队列创建之前运行初始化任务的竞争
            try:
                fut.result(timeout=2.0)
            except Exception:
                print("ModBusService.start() 没有在超时内完成；继续执行")
        except Exception:
            print("启动 ModBusService 失败")

    def _initialize_robot(self) -> None:
        """初始化机器人控制器状态（同步触发异步操作）"""
        # 用事件循环执行异步初始化操作
        async def init_tasks():
            try:
                await self.RobotEnable()
                await self.RobotReset()
                await self.SetControlMode(ControlMode.Calibration)
                await self.RobotSetParameters(self.robot_parameter)
                print("初始化成功！")
            except Exception as e:
                print(f"初始化机器人失败：{e}")

        if self._loop:
            asyncio.run_coroutine_threadsafe(init_tasks(), self._loop)

    # ------------------------------
    # 核心封装方法
    # ------------------------------
    # 重置机器人
    async def RobotReset(self) -> None:
        """重置机器人"""
        if not self.connected:
            raise RuntimeError("机器人未连接，无法执行重置操作")
        try:
            # 发送重置脉冲：先置位再复位
            await self._service.write_bool('Instructions.Reset_Execute', True)
            # await asyncio.sleep(0.1)  
            await self._service.write_bool('Instructions.Reset_Execute', False)
            print("机器人重置指令发送成功")
        except Exception as e:
            print(f"发送重置指令失败：{e}")
            raise

    # 使能机器人
    async def RobotEnable(self) -> None:
        """使能机器人"""
        if not self.connected:
            raise RuntimeError("机器人未连接，无法执行使能操作")
        
        try:
            await self._service.write_bool('Instructions.Power_On', True)
            print("机器人使能指令发送成功")
        except Exception as e:
            print(f"发送使能指令失败：{e}")
            raise
    
    # 失能机器人
    async def RobotDisable(self) -> None:
        """失能机器人"""
        if not self.connected:
            raise RuntimeError("机器人未连接，无法执行失能操作")
        
        try:
            await self._service.write_bool('Instructions.Power_On', False)
            print("机器人失能指令发送成功")
        except Exception as e:
            print(f"发送失能指令失败：{e}")
            raise

    # 设置机器人控制模式
    async def SetControlMode(self, mode: ControlMode) -> None:
        """设置机器人控制模式"""
        if not self.connected:
            raise RuntimeError("机器人未连接，无法设置控制模式")
        
        try:
            # 控制模式需转换为整数发送
            await self._service.write_uint('Parameters.Movement_Mode', int(mode.value))
            await asyncio.sleep(0.1)
            print(f"控制模式已设置为：{mode}")
        except Exception as e:
            print(f"设置控制模式失败：{e}")
            raise
    
    # 设置机器人参数
    async def RobotSetParameters(self, parameters: RobotParameters) -> None:
        """异步设置机器人参数（基于 ModBusService 接口）"""
        if not self.connected:
            raise RuntimeError("机器人未连接，无法设置参数")
        
        try:
            # 2. 写入全部关节 DH 参数（6个关节，每个关节4个参数）
            # 说明：每个关节包含4个参数（θ, d, a, α）
            dh_targets = [
                (i, j) for i in range(1, 7)  
                for j in range(1, 5)         
            ]

            for i, j in dh_targets:               
                try:
                    # 1. 校验索引范围
                    if not (1 <= i <= 6 and 1 <= j <= 4):
                        raise ValueError(f"索引超范围：i={i}（需1-6），j={j}（需1-4）")

                    # 2. 从DH_Parameters数组获取值（0-based索引：关节i-1，参数j-1）
                    key = "Parameters.DH_Parameters"
                    value = parameters.DH_Parameters[i-1][j-1]
                    if not isinstance(value, (int, float)):
                        raise TypeError(f"值类型错误：关节{i}参数{j}需为数字，实际为{type(value)}")
                    
                    await self._service.write_real(key, value, i, j)

                    await asyncio.sleep(0.05)  # 缩短间隔
                    print(f"已写入{key} 关节{i}参数{j}：{value}")

                except Exception as e:
                    print(f'写入关节{i}参数{j}失败：{e}\n')

            print("全部DH参数（6关节×4参数）写入完成")

            # 3. 写入 MoveJ 参考参数
            await self._service.write_real(
                'Parameters.MoveJ_Refference_Velocity', 
                parameters.MoveJReferenceVelocity
            )
            await self._service.write_real(
                'Parameters.MoveJ_Refference_Acceleration', 
                parameters.MoveJReferenceAcceleration
            )
            await self._service.write_real(
                'Parameters.MoveJ_Refference_Deceleration', 
                parameters.MoveJReferenceDeceleration
            )

            # 4. 写入 Jog 增量距离（关节 1-6）
            # 从InchDistance数组获取（0-based索引）
            # TODO 没有“Parameters.Inch_Distance”这个参数，需确认地址簿
            for i in range(1, 7):
                key = f'Parameters.Inch_Distance'
                value = parameters.InchDistance[i-1]  # 数组为0-based
                await self._service.write_real(key, value, i)

            # 5. 写入 Jog 速度（关节 1-6）
            # 从JointJogVelocity数组获取（0-based索引）
            for i in range(1, 7):
                key = f'Parameters.Joint_Jog_Velocity'
                value = parameters.JointJogVelocity[i-1]  # 数组为0-based
                await self._service.write_real(key, value, i)

            # 6. 写入 TCP（线性运动）参考参数（原MoveL相关参数，对应新属性）
            await self._service.write_real(
                'Parameters.TCP_Refference_Linear_Velocity', 
                parameters.TCPReferenceLinearVelocity
            )
            await self._service.write_real(
                'Parameters.TCP_Refference_Linear_Acceleration', 
                parameters.TCPReferenceLinearAcceleration
            )
            await self._service.write_real(
                'Parameters.TCP_Refference_Linear_Deceleration', 
                parameters.TCPReferenceLinearDeceleration
            )
            await self._service.write_real(
                'Parameters.TCP_Refference_Angular_Velocity', 
                parameters.TCPReferenceAngularVelocity
            )
            await self._service.write_real(
                'Parameters.TCP_Refference_Angular_Acceleration', 
                parameters.TCPReferenceAngularAcceleration
            )
            await self._service.write_real(
                'Parameters.TCP_Refference_Angular_Deceleration', 
                parameters.TCPReferenceAngularDeceleration
            )

            # 7. 写入关节参考速度（1-6）
            # 从JointReferenceVelocity数组获取（0-based索引）
            for i in range(1, 7):
                key = f'Parameters.Joint_Refference_Velocity'
                value = parameters.JointReferenceVelocity[i-1]  # 数组为0-based
                await self._service.write_real(key, value, i)

            # 8. 写入关节参考加速度（1-6）
            # 从JointReferenceAcceleration数组获取（0-based索引）
            for i in range(1, 7):
                key = f'Parameters.Joint_Refference_Acceleration'
                value = parameters.JointReferenceAcceleration[i-1]  # 数组为0-based
                await self._service.write_real(key, value, i)

            # 9. 写入关节参考加加速度（Jerk，1-6）
            # 从JointReferenceJerk数组获取（0-based索引）
            for i in range(1, 7):
                key = f'Parameters.Joint_Refference_Jerk'
                value = parameters.JointReferenceJerk[i-1]  # 数组为0-based
                await self._service.write_real(key, value, i)

        except Exception as e:
            print(f"参数设置失败：{e}")
            raise
    

    # TODO 功能核实 设置数字输出
    async def setDigitalOutput(self, index: int, value: bool) -> None:
        """设置数字输出（异步版）"""
        if not self.connected:
            raise RuntimeError("机器人未连接，无法设置数字输出")
        
        try:
            # 地址簿中数字输出的key格式假设为：DigitalOutputs.[index]
            # 需根据实际JSON地址簿调整key格式（如"Outputs.Digital[index]"）
            key = f'DigitalOutputs'
            await self._service.write_bool(key, value, index)
            await asyncio.sleep(0.5)

            print(f"数字输出 {index} 已设置为：{value}")
        except Exception as e:
            print(f"设置数字输出 {index} 失败：{e}")
            raise

    # 周期性读取机器人状态
    def ReadRobotStatus(self) -> None:
        """周期性读取机器人状态（定时器回调，同步触发异步任务）"""
        if self.connected and self._loop:
            # 定义异步读取任务
            async def async_read_status():
                try:
                    # 读取最新状态快照
                    new_status = await self._service.read_snapshot()
                    if new_status:
                        self.robot_status = new_status  # 更新状态
                        # print("机器人状态已更新")
                except Exception as e:
                    print(f"读取机器人状态失败：{e}")

            # 将异步任务提交到事件循环（非阻塞）
            asyncio.run_coroutine_threadsafe(async_read_status(), self._loop)
        # 重置定时器（保持周期性调用）
        self.timer = Timer(0.05, self.ReadRobotStatus)
        # Make timer a daemon so it won't block process exit on KeyboardInterrupt
        self.timer.daemon = True
        self.timer.start()

    # 获取机器人状态，并转换为JSON
    async def getRobotStatus(self) -> str:
        """获取当前机器人状态并转换为JSON"""
        if not self.connected:
            raise RuntimeError("机器人未连接，无法获取状态")
        
        try:
            # 主动读取最新状态，确保数据新鲜
            latest_status = await self._service.read_snapshot()
            if latest_status:
                self.robot_status = latest_status 

            # 定义内部函数：将状态对象转换为兼容旧版的结构
            def _normalize_robot_status(status_obj):
                from types import SimpleNamespace

                # 内部辅助函数：创建单个关节的命名空间
                def make_joint(i: int):
                    j = SimpleNamespace()
                    # 状态对象的列表是0索引，关节编号是1-6，需转换
                    j.ActualPosition = float(getattr(status_obj, 'JointActualPosition', [0]*6)[i-1]) if (i-1 < len(getattr(status_obj, 'JointActualPosition', []))) else 0.0
                    j.ActualVelocity = float(getattr(status_obj, 'JointActualVelocity', [0]*6)[i-1]) if (i-1 < len(getattr(status_obj, 'JointActualVelocity', []))) else 0.0
                    j.ActualAcceleration = float(getattr(status_obj, 'JointActualAcceleration', [0]*6)[i-1]) if (i-1 < len(getattr(status_obj, 'JointActualAcceleration', []))) else 0.0
                    j.ActualCurrent = float(getattr(status_obj, 'JointActualCurrent', [0]*6)[i-1]) if (i-1 < len(getattr(status_obj, 'JointActualCurrent', []))) else 0.0
                    j.ActualTorque = float(getattr(status_obj, 'JointActualTorque', [0]*6)[i-1]) if (i-1 < len(getattr(status_obj, 'JointActualTorque', []))) else 0.0
                    return j

                # 构建兼容旧版的状态结构
                legacy = SimpleNamespace()
                # 1. 关节信息（J1-J6）
                joints_ns = SimpleNamespace()
                for idx in range(1, 7):
                    setattr(joints_ns, f'J{idx}', make_joint(idx))
                legacy.Joints = joints_ns
                
                # 2. 管理信息（运动状态、使能状态等）
                manage = SimpleNamespace()
                manage.Moving = bool(getattr(status_obj, 'Moving', False))
                manage.Enabled = bool(getattr(status_obj, 'PowerOn', False))
                manage.Mode = int(getattr(status_obj, 'Mode', 0)) if getattr(status_obj, 'Mode', None) is not None else 0
                manage.Error = bool(getattr(status_obj, 'Error', False))
                legacy.Manage = manage
                
                # 3. TCP位姿信息
                tcp = SimpleNamespace()
                tp = getattr(status_obj, 'TcpPose', [0.0]*6)
                try:
                    tcp.X = float(tp[0])
                    tcp.Y = float(tp[1])
                    tcp.Z = float(tp[2])
                    tcp.Roll = float(tp[3])
                    tcp.Pitch = float(tp[4])
                    tcp.Yaw = float(tp[5])
                except Exception:
                    tcp.X = tcp.Y = tcp.Z = tcp.Roll = tcp.Pitch = tcp.Yaw = 0.0
                legacy.TcpPose = tcp
                
                # 4. 新增：AirLock（气锁状态，适配 transform_json 的需求）
                airlock_data = getattr(status_obj, 'AirLock', {})  
                # 确保 airlock 是可序列化的结构（根据实际业务补充字段，如是否锁止、压力等）
                airlock = SimpleNamespace(
                    locked=bool(getattr(airlock_data, 'locked', False)),  # 是否锁止
                    pressure=float(getattr(airlock_data, 'pressure', 0.0))  # 气压值
                )
                legacy.AirLock = airlock

                # 提供to_dict方法用于序列化
                def to_dict():
                    return {
                        'Joints': {f'J{i}': vars(getattr(legacy.Joints, f'J{i}')) for i in range(1,7)},
                        'Manage': vars(legacy.Manage),
                        'TcpPose': vars(legacy.TcpPose),
                        'AirLock': vars(legacy.AirLock)
                    }
                legacy.to_dict = to_dict
                return legacy

            # 处理状态字典的生成（优先使用自带to_dict，否则标准化）
            to_dict_fn = getattr(self.robot_status, 'to_dict', None)
            if callable(to_dict_fn):
                status_dict = to_dict_fn()
            else:
                legacy = _normalize_robot_status(self.robot_status)
                status_dict = legacy.to_dict()

            # [调试]打印字典
            # pprint(status_dict)

            # 转换为JSON并返回
            transformed = transform_json(status_dict)
            
            return json.dumps(transformed, indent=4)
        
        except Exception as e:
            print(f"获取机器人状态失败：{e}")
            raise

    # 读取机器人当前关节角度
    async def getCurrentJointAngles(self) -> np.ndarray:
        """
        获取当前机器人6个关节的角度，并转换为弧度 (rad)。
        专为高频 PBVS 实时控制优化，去除了冗余状态和 JSON 转换。
        :return: numpy array 包含 6 个关节的弧度值 [q1, q2, q3, q4, q5, q6]
        """
        if not self.connected:
            raise RuntimeError("机器人未连接，无法获取关节角度")

        try:
            # 1. 主动读取最新状态快照，确保数据新鲜
            latest_status = await self._service.read_snapshot()

            if not latest_status:
                raise ValueError("读取到的机器人状态为空")

            # 2. 仅提取关节位置列表 (单位：弧度)
            # 如果底层对象没有该属性，则提供默认的 6 个 0.0 防止报错
            joints_rad = getattr(latest_status, 'JointActualPosition', [0.0] * 6)

            # 安全检查：确保拿到的是 6 个元素的列表/元组
            if len(joints_rad) < 6:
                joints_deg = list(joints_rad) + [0.0] * (6 - len(joints_rad))
            elif len(joints_rad) > 6:
                joints_deg = joints_rad[:6]

            # 3. 转换为浮点型 numpy 数组
            joints_rad_np = np.array(joints_rad, dtype=float)

            # 4. 将获取得到的关节角度由弧度转化为度，用于显示用
            joints_deg_np = np.rad2deg(joints_rad_np)

            return joints_rad_np

        except Exception as e:
            print(f"获取机器人关节角度失败：{e}")
            raise


    # TODO 关节正向连续点动（速度模式）
    async def JogForward(self, index: int, value: bool) -> None:
        """关节正向连续点动（速度模式）"""
        if not self.connected:
            raise RuntimeError("机器人未连接，无法执行正向点动")
        
        try:
            # 1. 设置控制模式为关节点动模式
            await self.SetControlMode(ControlMode.JointJog)

            # 2. 设置点动模式为速度模式（连续点动）
            await self._service.write_uint('Parameters.Movement_Mode', int(JogMode.VelocityJog.value))

            # 3. 发送正向点动指令（value为True启动，False停止）
            key = f'Instructions.Joint_Inch_Forward'
            await self._service.write_bool(key, value)
            await asyncio.sleep(0.1) 
            print(f"关节 {index} 正向连续点动已{'启动' if value else '停止'}")
        except Exception as e:
            print(f"关节 {index} 正向连续点动失败：{e}")
            raise

    # TODO 功能核实 关节正向增量点动（距离模式）        
    async def InchForward(self, index: int, value: bool) -> None:
        """关节正向增量点动（距离模式）"""
        if not self.connected:
            raise RuntimeError("机器人未连接，无法执行正向增量点动")
        
        try:
            # 1. 设置控制模式为关节点动模式
            await self.SetControlMode(ControlMode.JointJog)

            # 2. 设置点动模式为距离模式（增量点动）
            await self._service.write_uint('Parameters.Movement_Mode', int(JogMode.DistanceJog.value))

            # 3. 发送正向增量点动指令（value为True触发，False复位）
            key = f'Instructions.Joint_Inch_Forward'
            await self._service.write_bool(key, value, index)
            await asyncio.sleep(0.1)
            print(f"关节 {index} 正向增量点动已{'触发' if value else '复位'}")
        except Exception as e:
            print(f"关节 {index} 正向增量点动失败：{e}")
            raise

    # TODO 功能核实 关节反向连续点动（速度模式）
    async def JogBackward(self, index: int, value: bool) -> None:
        """关节反向连续点动（速度模式）"""
        if not self.connected:
            raise RuntimeError("机器人未连接，无法执行反向点动")
        
        try:
            # 1. 设置控制模式为关节点动模式
            await self.SetControlMode(ControlMode.JointJog)

            # 2. 设置点动模式为速度模式（连续点动）
            await self._service.write_uint('Parameters.Movement_Mode', int(JogMode.VelocityJog.value))

            # 3. 发送反向点动指令
            # TODO 索引校验 （0-based or 1-based)
            key = f'Instructions.Joint_Inch_Backward'
            await self._service.write_bool(key, value, index)
            await asyncio.sleep(0.1)
            print(f"关节 {index} 反向连续点动已{'启动' if value else '停止'}")
        except Exception as e:
            print(f"关节 {index} 反向连续点动失败：{e}")
            raise

    # 关节反向增量点动（距离模式）
    async def InchBackward(self, index: int, value: bool) -> None:
        """关节反向增量点动（距离模式）"""
        if not self.connected:
            raise RuntimeError("机器人未连接，无法执行反向增量点动")
        
        try:
            # 1. 设置控制模式为关节点动模式
            await self.SetControlMode(ControlMode.JointJog)

            # 2. 设置点动模式为距离模式（增量点动）
            await self._service.write_uint('Parameters.Movement_Mode', int(JogMode.DistanceJog.value))

            # 3. 发送反向增量点动指令
            # TODO 索引校验 （0-based or 1-based)
            key = f'Instructions.Joint_Inch_Backward'
            await self._service.write_bool(key, value, index)
            await asyncio.sleep(0.1)
            print(f"关节 {index} 反向增量点动已{'触发' if value else '复位'}")
        except Exception as e:
            print(f"关节 {index} 反向增量点动失败：{e}")
            raise

    # 关节点动统一接口
    async def jointJog(self, axis: str, direction: int, type: str, isStart: bool) -> None:
        """
        关节点动控制（统一接口）
        :param axis: 关节名称(如"J1"、"J2")
        :param direction: 方向(1:正向，-1:反向)
        :param type: 模式（"speed":速度模式，"distance":距离模式)
        :param isStart: 是否启动(True:启动,False:停止)
        """
        if not self.connected:
            raise RuntimeError("机器人未连接，无法执行点动")
        
        try:
            # 1. 转换关节名称为索引
            axisIndex = get_joint_index(axis)
            if axisIndex < 1 or axisIndex > 6:  # 假设关节索引1-6
                raise ValueError(f"无效的关节名称：{axis}")
            
            # 2. 设置点动模式（速度/距离）
            jog_mode = JogMode.VelocityJog if type == "speed" else JogMode.DistanceJog
            await self._service.write_uint('Jog.Mode', int(jog_mode.value)) # TODO 地址确认
            
            # 3. 设置控制模式为关节点动
            await self.SetControlMode(ControlMode.JointJog)
            
            # 4. 根据方向发送点动指令
            if direction == 1:
                await self.JogForward(axisIndex, isStart)  
            elif direction == -1:
                await self.JogBackward(axisIndex, isStart) 
            else:
                raise ValueError(f"无效的方向：{direction}（需为1或-1）")

            print(f"关节 {axis} 点动（{type}模式）已{'启动' if isStart else '停止'}，方向：{direction}")
        except Exception as e:
            print(f"关节点动失败：{e}")
            raise

    # 关节空间运动(发送的关节角度单位是弧度)
    async def MoveJ(self, joint_positions: List[float]) -> None:
        """
        关节空间运动
        :param joint_positions: 目标关节位置列表（[J1, J2, J3, J4, J5, J6]）
        """
        if not self.connected:
            raise RuntimeError("机器人未连接，无法执行MoveJ")
        if len(joint_positions) != 6:
            raise ValueError("关节位置列表必须包含6个元素（J1-J6）")
        
        try:
            # 1. 写入目标关节位置
            for i in range(6):
                key = f'Parameters.Joint_Target_Position'  # 1-based index
                await self._service.write_real(key, joint_positions[i], i+1)
            
            # 2. 设置控制模式为关节运动模式
            await self.SetControlMode(ControlMode.MoveJoint)
            
            # 3. 触发关节运动执行 
            # TODO 核对key，key名不确定
            await self._service.write_bool('Instructions.TCP_Move_Joint_Execute', True)
            await asyncio.sleep(0.1)  # 等待触发生效
            # （可选）复位触发信号（根据设备需求）
            await self._service.write_bool('Instructions.TCP_Move_Joint_Execute', False)

            print("MoveJ 指令已发送")
        except Exception as e:
            print(f"MoveJ 执行失败：{e}")
            raise

    # 关节空间运动
    async def MoveJWait(self, joint_positions: List[float]) -> None:
        """
        关节空间运动（阻塞，等待完成）
        :param joint_positions: 目标关节位置列表（[J1, J2, J3, J4, J5, J6]）
        """
        # print("MoveJWait 执行")
        if not self.connected:
            raise RuntimeError("机器人未连接，无法执行MoveJWait")
        
        try:
            # 1. 发送运动指令
            await self.MoveJ(joint_positions)
            
            # 2. 等待运动完成（循环检查状态）
            while True:
                # 异步获取最新状态
                self.robot_status = await self._service.read_snapshot()
                if not getattr(self.robot_status, 'Moving', False):  # 运动停止
                    break
                await asyncio.sleep(0.1)  
                
            # 3. 运动完成后检查误差（安全地获取关节实际位置，兼容多种状态对象形状）
            # 先尝试从 JointActualPosition 列表获取
            positions = getattr(self.robot_status, 'JointActualPosition', None)
            if positions is None:
                # 回退到 legacy 形状（Joints.J1..J6）
                joints = getattr(self.robot_status, 'Joints', None)
                if joints is not None:
                    try:
                        positions = [
                            float(getattr(joints.J1, 'ActualPosition', 0.0)),
                            float(getattr(joints.J2, 'ActualPosition', 0.0)),
                            float(getattr(joints.J3, 'ActualPosition', 0.0)),
                            float(getattr(joints.J4, 'ActualPosition', 0.0)),
                            float(getattr(joints.J5, 'ActualPosition', 0.0)),
                            float(getattr(joints.J6, 'ActualPosition', 0.0)),
                        ]
                    except Exception:
                        positions = [0.0] * 6
                else:
                    # 默认 6 个 0.0
                    positions = [0.0] * 6
            else:
                # 确保为长度为6的浮点列表
                positions = list(positions)[:6] + [0.0] * max(0, 6 - len(positions))
                positions = [float(x) for x in positions]

            error = Utils.caculate_joints_error(joint_positions, positions)
            
            if error < 0.1:
                print(f"MoveJ 完成，到达目标位置（误差：{error}）")
                print("已到达目标位置")
            else:
                print(f"MoveJ 误差过大：{error}")
                raise Exception(f"未到达目标位置（误差：{error}）")
                
        except Exception as e:
            print(f"MoveJWait 执行失败：{e}")
            raise

    # 笛卡尔空间运动（异步） 
    async def MoveL(self, pose: List[float]) -> None:
        """
        笛卡尔空间线性运动（非阻塞）
        :param pose: 目标位姿列表（[X, Y, Z, Roll, Pitch, Yaw]）
        """
        if not self.connected:
            raise RuntimeError("机器人未连接，无法执行MoveL")
        if len(pose) != 6:
            raise ValueError("位姿列表必须包含6个元素（X,Y,Z,Roll,Pitch,Yaw）")
        
        try:
            # 1. 写入目标笛卡尔位姿（Target[i]，i=0-5）
            for i in range(6):
                key = f'Parameters.TCP_Target_Pose'
                await self._service.write_real(key, pose[i], i+1) 
            
            # 2. 设置控制模式为线性运动模式
            await self.SetControlMode(ControlMode.MoveLinear)
            
            # 3. 触发线性运动执行
            await self._service.write_bool('Instructions.TCP_Move_Linear_Execute', True)
            await asyncio.sleep(0.1)  # 等待触发生效
            # （可选）复位触发信号
            await self._service.write_bool('Instructions.TCP_Move_Linear_Execute', False)

            print("MoveL 指令已发送")
        except Exception as e:
            print(f"MoveL 执行失败：{e}")
            raise

    # 笛卡尔空间线性运动
    async def MoveLWait(self, pose: List[float]) -> None:
        """
        笛卡尔空间线性运动（阻塞，等待完成）
        :param pose: 目标位姿列表（[X, Y, Z, Roll, Pitch, Yaw]）
        """
        if not self.connected:
            raise RuntimeError("机器人未连接，无法执行MoveLWait")
        
        try:
            # 1. 发送运动指令
            await self.MoveL(pose)
            
            # 2. 等待运动完成（循环检查状态）
            await asyncio.sleep(1.0) 
            while True:
                # 异步获取最新状态
                self.robot_status = await self._service.read_snapshot()
                if not getattr(self.robot_status, 'Moving', False): 
                    break
                await asyncio.sleep(0.5)  
                
            # 3. 运动完成后检查误差
            # 安全地获取 TcpPose，避免 self.robot_status 为 None 或 TcpPose 为 None 时索引出错
            tp = getattr(self.robot_status, 'TcpPose', [])
            if isinstance(tp, (list, tuple)):
                vals = list(tp)[:6] + [0.0] * max(0, 6 - len(tp))
                actual_pose = [float(v) for v in vals]
            else:
                try:
                    actual_pose = [
                        float(getattr(tp, 'X', 0.0)),
                        float(getattr(tp, 'Y', 0.0)),
                        float(getattr(tp, 'Z', 0.0)),
                        float(getattr(tp, 'Roll', 0.0)),
                        float(getattr(tp, 'Pitch', 0.0)),
                        float(getattr(tp, 'Yaw', 0.0)),
                    ]
                except Exception:
                    actual_pose = [0.0] * 6
            position_error, rotation_error = Utils.caculate_cartesian_error(pose, actual_pose)
            
            if position_error < 1 and rotation_error < 1:
                print(f"MoveL 完成，到达目标位置（位置误差：{position_error}，姿态误差：{rotation_error}）")
                print("已到达目标位置")
            else:
                error_msg = []
                if position_error >= 1:
                    error_msg.append(f"位置误差过大：{position_error}")
                if rotation_error >= 1:
                    error_msg.append(f"姿态误差过大：{rotation_error}")
                print("; ".join(error_msg))
                raise Exception("; ".join(error_msg))
                
        except Exception as e:
            print(f"MoveLWait 执行失败：{e}")
            raise

    # 关节绝对运动
    async def MoveAbs(self, joint_index: int, target_position: float) -> None:
        """
        单关节绝对运动（非阻塞）
        :param joint_index: 关节索引（1-6）
        :param target_position: 目标绝对位置
        """
        if not self.connected:
            raise RuntimeError("机器人未连接，无法执行关节绝对运动")
        if not (1 <= joint_index <= 6):
            raise ValueError(f"无效的关节索引：{joint_index}（需为1-6）")
        
        try:
            # 1. 写入目标绝对位置 
            # TODO 核对key
            key = f'Parameters.Joint_Target_Position'
            await self._service.write_real(key, target_position, joint_index)
            
            # 2. 设置控制模式为关节绝对运动模式
            await self.SetControlMode(ControlMode.JointMoveAbs)

            # 3. 触发单关节绝对运动执行
            execute_key = f'Instructions.Joint_MoveABS_Execute'
            await self._service.write_bool(execute_key, True)
            await asyncio.sleep(0.1)  # 等待触发生效
            await self._service.write_bool(execute_key, False)  # 复位触发信号

            print(f"关节 {joint_index} 绝对运动指令已发送（目标位置：{target_position}）")
        except Exception as e:
            print(f"关节 {joint_index} 绝对运动执行失败：{e}")
            raise

    # 关节绝对运动        
    async def MoveABSWait(self, joint_index: int, target_position: float) -> None:
        """
        单关节绝对运动（阻塞，等待完成）
        :param joint_index: 关节索引（1-6）
        :param target_position: 目标绝对位置
        """
        if not self.connected:
            raise RuntimeError("机器人未连接，无法执行关节绝对运动（等待完成）")
        if not (1 <= joint_index <= 6):
            raise ValueError(f"无效的关节索引：{joint_index}（需为1-6）")
        
        try:
            # 1. 发送关节绝对运动指令
            await self.MoveAbs(joint_index, target_position)
            
            # 2. 等待运动完成（循环检查状态）
            while True:
                current_status = await self._service.read_snapshot()
                if not current_status:
                    raise RuntimeError("读取运动状态失败")
                if not current_status.Moving:  # 运动停止
                    break
                await asyncio.sleep(0.1)  
            
            # 3. 运动完成后检查误差（关节索引从1开始，列表从0开始，需减1）
            actual_position = current_status.JointActualPosition[joint_index - 1]
            error = abs(target_position - actual_position)
            
            if error < 0.1:
                print(f"关节 {joint_index} 绝对运动完成，到达目标位置（误差：{error}）")
            else:
                print(f"关节 {joint_index} 绝对运动误差过大：{error}")
                raise Exception(f"关节 {joint_index} 未到达目标位置（误差：{error}）")
                
        except Exception as e:
            print(f"关节 {joint_index} 绝对运动（等待完成）执行失败：{e}")
            raise

    # TODO TCP速度控制
    async def Move_tcp_velocity(self, velocity_vector: List[float]) -> None:
        """
        TCP速度控制（设置TCP的速度向量）
        :param velocity_vector: 速度向量列表（[vx, vy, vz, vRoll, vPitch, vYaw]）
        """
        if not self.connected:
            raise RuntimeError("机器人未连接，无法执行TCP速度控制")
        if len(velocity_vector) != 6:
            raise ValueError("速度向量必须包含6个元素（vx, vy, vz, vRoll, vPitch, vYaw）")
        
        try:
            # TODO TCP.Velocity
            # # 假设地址簿中TCP速度控制的key为'TCP.Velocity[i]'（i=0-5）
            # for i in range(6):
            #     key = f'TCP.Velocity[{i}]'
            #     await self._service.write_real(key, velocity_vector[i])
            pass

            print("TCP速度向量已设置")
        except Exception as e:
            print(f"TCP速度控制执行失败：{e}")
            raise


    # 关节绝对运动停止
    async def MoveAbsStop(self, joint_index: int) -> None:
        """
        停止单关节绝对运动
        :param joint_index: 关节索引（1-6）
        """
        if not self.connected:
            raise RuntimeError("机器人未连接，无法停止关节绝对运动")
        if not (1 <= joint_index <= 6):
            raise ValueError(f"无效的关节索引：{joint_index}（需为1-6）")
        
        try:
            # 假设停止信号
            stop_key = f'Instructions.Joint_Stop_Execute'
            await self._service.write_bool(stop_key, True)
            await asyncio.sleep(0.1)  
            await self._service.write_bool(stop_key, False)  

            print(f"关节 {joint_index} 绝对运动已停止")
        except Exception as e:
            print(f"停止关节 {joint_index} 绝对运动失败：{e}")
            raise

    # 关节绝对运动统一接口
    async def JointMoveAbs(self, joint_name: str, isStop: bool, target_position: Optional[float] = None) -> None:
        """
        关节绝对运动统一控制（启动/停止）
        :param joint_name: 关节名称（如"J1"、"J2"）
        :param isStop: 是否停止（True:停止，False:启动）
        :param target_position: 目标位置（isStop=False时必传）
        """
        joint_index = get_joint_index(joint_name)  # 复用关节名称转索引的工具函数
        if not (1 <= joint_index <= 6):
            raise ValueError(f"无效的关节名称：{joint_name}")
        
        if not isStop:
            if target_position is None:
                raise ValueError("启动运动时必须指定目标位置")
            await self.MoveAbs(joint_index, target_position)
        else:
            await self.MoveAbsStop(joint_index)


    # 机器人全局控制
    async def RobotStop(self) -> None:
        """停止机器人所有运动"""
        if not self.connected:
            raise RuntimeError("机器人未连接，无法执行停止操作")
        
        try:
            await self._service.write_bool('Instructions.Stop_Execute', True)
            await asyncio.sleep(0.1)
            await self._service.write_bool('Instructions.Stop_Execute', False)
            print("机器人已停止所有运动")
        except Exception as e:
            print(f"机器人停止失败：{e}")
            raise

    # 控制模式激活
    async def activateJointMoveAbs(self) -> None:
        """激活关节绝对运动模式"""
        await self.SetControlMode(ControlMode.JointMoveAbs)

    async def activateMoveLinear(self) -> None:
        """激活线性运动模式"""
        await self.SetControlMode(ControlMode.MoveLinear)

    async def activateMoveJoint(self) -> None:
        """激活关节运动模式"""
        await self.SetControlMode(ControlMode.MoveJoint)

    async def activateJointJog(self) -> None:
        """激活关节点动模式"""
        await self.SetControlMode(ControlMode.JointJog)

    def stop(self) -> None:
        """停止机器人服务、事件循环及相关资源"""
        if not self.connected:
            print("机器人未连接，无需停止服务")
            return
        try:
            # 1. 取消状态读取定时器
            if self.timer:
                self.timer.cancel()
                print("已取消状态读取定时器")

            # 2. 停止ModBusService
            if self._service and self._loop:
                try:
                    # 提交停止服务的异步任务并等待完成
                    stop_fut = asyncio.run_coroutine_threadsafe(self._service.stop(), self._loop)
                    stop_fut.result(timeout=5.0) 
                    print("ModBusService已停止")
                except asyncio.TimeoutError:
                    print("停止ModBusService超时")
                except Exception as e:
                    print(f"停止ModBusService时发生错误: {e}")

            # 3. 停止事件循环并等待线程结束
            if self._loop:
                self._loop.call_soon_threadsafe(self._loop.stop)
                print("已请求事件循环停止")

            if self._loop_thread and self._loop_thread.is_alive():
                self._loop_thread.join(timeout=2.0)
                if self._loop_thread.is_alive():
                    print("事件循环线程未能正常退出")
                else:
                    print("事件循环线程已退出")

            # 4. 标记连接状态为断开
            self.connected = False
            print("机器人服务已成功停止")

        except Exception as e:
            print(f"停止机器人服务时发生未预期错误: {e}")
