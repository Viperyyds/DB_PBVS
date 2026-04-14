import time
from time import sleep
import threading
import asyncio
from typing import Optional, List, Union
from threading import Timer

from Core.Basic import *
from Communication.ModBusService import ModBusService, RobotStatus
from Communication.ModBusCommunicator import ModBusCommunicator
from Communication.CompactEntry import AddressBook

import logging

from Util import *
import json
from pprint import pprint
from Util import *
from Core.gripper import Gripper

LOGGER = logging.getLogger(__name__)

class RobotCore:
    def __init__(self,target_ip:str):
        # 创建底层同步 communicator
        self.communicator = ModBusCommunicator(target_ip, 502, unit_id=1, client=None)

        # 加载地址簿
        self._address_book = self._load_address_book()

        # 创建 ModBusService（async 管理 I/O 队列）
        self._service = ModBusService(self.communicator, self._address_book or {})

        # 事件循环与后台线程
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread = None

        # 机器人状态与参数
        self.robot_parameter = RobotParameters()
        self.robot_status = RobotStatus()
        self.gripper = Gripper("COM4")
        self.timer = Timer(0.05, self.ReadRobotStatus)

        # 连接状态标记
        self.connected = False

        # 尝试连接并初始化
        self._initialize_connection()

    def _load_address_book(self) -> Optional[dict]:
        """加载地址簿的工具方法"""
        # TODO 按需修改json文件路径
        try_paths = [
            'Communication/modbus_address_book.compact_win.json',
            'modbus_address_book.compact_win.json'
        ]
        for path in try_paths:
            try:
                return AddressBook.load(path)
            except Exception as e:
                LOGGER.warning(f"加载地址簿失败（路径：{path}）：{e}")
        return None

    def _initialize_connection(self) -> None:
        """初始化连接并启动服务"""
        try:
            self.connected = self.communicator.connect()
        except Exception as e:
            LOGGER.error(f"连接机器人失败：{e}")
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

    def _initialize_robot(self) -> None:
        """初始化机器人控制器状态（同步触发异步操作）"""
        # 用事件循环执行异步初始化操作
        async def init_tasks():
            try:
                await self.RobotEnable()
                await self.RobotReset()
                await self.SetControlMode(ControlMode.Idel)
                # 调用新的异步参数设置方法（替代旧的同步调用）
                await self.RobotSetParameters(self.robot_parameter)
                print("初始化成功！")
            except Exception as e:
                LOGGER.error(f"初始化机器人失败：{e}")

        if self._loop:
            asyncio.run_coroutine_threadsafe(init_tasks(), self._loop)

   
    # ------------------------------
    # 核心封装方法
    # ------------------------------
    # 重置机器人
    async def RobotReset(self) -> None:
        """重置机器人（异步版）"""
        if not self.connected:
            raise RuntimeError("机器人未连接，无法执行重置操作")
        
        try:
            # 发送重置脉冲：先置位再复位
            await self._service.write_bool('Instructions.Reset_Execute', True)
            await asyncio.sleep(0.1)  # 异步等待，避免阻塞事件循环
            await self._service.write_bool('Instructions.Reset_Execute', False)
            LOGGER.info("机器人重置指令发送成功")
        except Exception as e:
            LOGGER.error(f"发送重置指令失败：{e}")
            raise  # 向上层抛出异常，方便调用者处理
    
    # 使能机器人
    async def RobotEnable(self) -> None:
        """使能机器人（异步版）"""
        if not self.connected:
            raise RuntimeError("机器人未连接，无法执行使能操作")
        
        try:
            await self._service.write_bool('Instructions.Power_On', True)
            LOGGER.info("机器人使能指令发送成功")
        except Exception as e:
            LOGGER.error(f"发送使能指令失败：{e}")
            raise
    
    # 失能机器人
    async def RobotDisable(self) -> None:
        """失能机器人（异步版）"""
        if not self.connected:
            raise RuntimeError("机器人未连接，无法执行失能操作")
        
        try:
            await self._service.write_bool('Instructions.Power_On', False)
            LOGGER.info("机器人失能指令发送成功")
        except Exception as e:
            LOGGER.error(f"发送失能指令失败：{e}")
            raise

    # 设置机器人控制模式
    async def SetControlMode(self, mode: ControlMode) -> None:
        """设置机器人控制模式（异步版）"""
        if not self.connected:
            raise RuntimeError("机器人未连接，无法设置控制模式")
        
        try:
            # 控制模式需转换为整数发送
            await self._service.write_uint('Parameters.Movement_Mode', int(mode))
            await asyncio.sleep(0.1)  # 等待模式切换生效
            LOGGER.info(f"控制模式已设置为：{mode}")
        except Exception as e:
            LOGGER.error(f"设置控制模式失败：{e}")
            raise
    
    # 设置机器人参数
    async def RobotSetParameters(self, parameters: RobotParameters) -> None:
        """异步设置机器人参数（基于 ModBusService 接口）"""
        if not self.connected:
            raise RuntimeError("机器人未连接，无法设置参数")
        
        try:
            # 1. 使能参数写入（地址簿中对应 key，需与实际配置一致）
            await self._service.write_bool('Parameters.Write_Enable', True)
            LOGGER.info("已使能参数写入权限")

            # 2. 写入关节 DH 参数（仅需 D1、D3、D6，对应关节1、3、6的d参数）
            # 说明：d参数是每个关节的第3个参数（索引3），对应结构 [关节号][3]
            dh_targets = [
                (1, 'DH_1_d'),   # D1：关节1的d参数
                (3, 'DH_3_d'),   # D3：关节3的d参数
                (6, 'DH_6_d')    # D6：关节6的d参数
            ]

            for joint_idx, param_name in dh_targets:
                # 构造地址簿key：关节joint_idx的第3个参数（d参数） 
                # TODO 验证
                key = f'Parameters.DH_Parameters[{joint_idx}][3]'
                # 从parameters中获取对应值（如DH_1_d对应关节1的d参数）
                value = getattr(parameters, param_name)
                # 写入参数
                await self._service.write_real(key, value)
                LOGGER.debug(f"已写入关节{joint_idx}的d参数（{param_name}）：{value}")

            LOGGER.info("DH参数（D1、D3、D6）写入完成")

            # # 2. 写入关节 DH 参数（1..6关节 × 1..4参数，顺序：a→alpha→d→theta）
            # for joint_idx in range(1, 7):  # 关节1到6
            #     # 第1个参数：a
            #     key_a = f'Parameters.DH_Parameters[{joint_idx}][1]'
            #     value_a = getattr(parameters, f'DH_{joint_idx}_a')  # 从parameters取对应值
            #     await self._service.write_real(key_a, value_a)
                
            #     # 第2个参数：alpha
            #     key_alpha = f'Parameters.DH_Parameters[{joint_idx}][2]'
            #     value_alpha = getattr(parameters, f'DH_{joint_idx}_alpha')
            #     await self._service.write_real(key_alpha, value_alpha)
                
            #     # 第3个参数：d
            #     key_d = f'Parameters.DH_Parameters[{joint_idx}][3]'
            #     value_d = getattr(parameters, f'DH_{joint_idx}_d')
            #     await self._service.write_real(key_d, value_d)
                
            #     # 第4个参数：theta
            #     key_theta = f'Parameters.DH_Parameters[{joint_idx}][4]'
            #     value_theta = getattr(parameters, f'DH_{joint_idx}_theta')
            #     await self._service.write_real(key_theta, value_theta)

            # LOGGER.info("6组关节的DH参数（a/alpha/d/theta）写入完成")
          

            # 3. 写入 MoveJ 参考参数
            await self._service.write_real('Parameters.MoveJ.Reference_Velocity', parameters.MoveJ_Refference_Velocity)
            await self._service.write_real('Parameters.MoveJ.Reference_Acceleration', parameters.MoveJ_Refference_Acceleration)
            await self._service.write_real('Parameters.MoveJ.Reference_Deceleration', parameters.MoveJ_Refference_Deceleration)

            # 4. 写入 Jog 增量距离（关节 1-6）
            for i in range(1, 7):
                key = f'Parameters.Inch_Distance{i}'
                value = getattr(parameters, f'Jog_IncDistance_{i}')
                await self._service.write_real(key, value)

            # 5. 写入 Jog 速度（关节 1-6）
            for i in range(1, 7):
                key = f'Parameters.Joint_Jog_Velocity{i}'
                value = getattr(parameters, f'Jog_Velocity_{i}')
                await self._service.write_real(key, value)

            # 6. 写入 MoveL 参考参数
            # TODO json文件中没有MoveL相关参数，先仿照MoveJ添加
            await self._service.write_real('Parameters.MoveL_Refference_Velocity', parameters.MoveL_Reference_Linear_Velocity)
            await self._service.write_real('Parameters.MoveL_Refference_Acceleration', parameters.MoveL_Reference_Linear_Acceleration)
            await self._service.write_real('Parameters.MoveL_Refference_Deceleration', parameters.MoveL_Reference_Linear_Deceleration)
            await self._service.write_real('Parameters.MoveL_Refference_Angular_Velocity', parameters.MoveL_Reference_Angular_Velocity)
            await self._service.write_real('Parameters.MoveL_Refference_Angular_Acceleration', parameters.MoveL_Reference_Angular_Acceleration)
            await self._service.write_real('Parameters.MoveL_Refference_Angular_Deceleration', parameters.MoveL_Reference_Angular_Deceleration)

            # 7. 写入关节参考速度（1-6）
            for i in range(1, 7):
                key = f'Parameters.Joint_Refference_Velocity{i}'
                value = getattr(parameters, f'Joint{i}_Reference_Velocity')
                await self._service.write_real(key, value)

            # 8. 写入关节参考加速度（1-6）
            for i in range(1, 7):
                key = f'Parameters.Joint_Refference_Acceleration{i}'
                value = getattr(parameters, f'Joint{i}_Reference_Acceleration')
                await self._service.write_real(key, value)

            # 9. 写入关节参考加加速度（Jerk，1-6）
            for i in range(1, 7):
                key = f'Parameters.Joint_Refference_Jerk{i}'
                value = getattr(parameters, f'Joint{i}_Reference_Jerk')
                await self._service.write_real(key, value)

            # 10. 禁用参数写入
            # TODO 没有此参数
            # await self._service.write_bool('Parameters.Write_Enable', False)
            # LOGGER.info("机器人参数设置完成，已禁用参数写入权限")

        except Exception as e:
            LOGGER.error(f"参数设置失败：{e}", exc_info=True)
            # 异常时确保关闭参数写入权限
            try:
                await self._service.write_bool('Parameters.Write_Enable', False)
            except Exception:
                LOGGER.warning("参数写入失败后，关闭写入权限也失败")
            raise  # 向上层抛出异常
    
    # TODO 设置数字输出
    async def setDigitalOutput(self, index: int, value: bool) -> None:
        """设置数字输出（异步版）"""
        if not self.connected:
            raise RuntimeError("机器人未连接，无法设置数字输出")
        
        try:
            # 地址簿中数字输出的key格式假设为：DigitalOutputs.[index]
            # 需根据实际JSON地址簿调整key格式（如"Outputs.Digital[index]"）
            # TODO 修改变量名
            key = f'DigitalOutputs.{index}'
            await self._service.write_bool(key, value)
            # 保留原逻辑中的延迟（异步等待，避免阻塞事件循环）
            await asyncio.sleep(0.5)
            LOGGER.info(f"数字输出 {index} 已设置为：{value}")
        except Exception as e:
            LOGGER.error(f"设置数字输出 {index} 失败：{e}")
            raise

    # 周期性读取机器人状态
    def ReadRobotStatus(self) -> None:
        """周期性读取机器人状态（定时器回调，同步触发异步任务）"""
        if self.connected and self._loop:
            # 定义异步读取任务
            async def async_read_status():
                try:
                    # 调用 ModBusService 读取最新状态快照
                    new_status = await self._service.read_snapshot()
                    if new_status:
                        self.robot_status = new_status  # 更新状态
                        LOGGER.debug("机器人状态已更新")
                except Exception as e:
                    LOGGER.warning(f"读取机器人状态失败：{e}")

            # 将异步任务提交到事件循环（非阻塞）
            asyncio.run_coroutine_threadsafe(async_read_status(), self._loop)
        # 重置定时器（保持周期性调用）
        self.timer = Timer(0.05, self.ReadRobotStatus)
        self.timer.start()

    # # 获取机器人状态，并转换为JSON
    # async def getRobotStatus(self) -> str:
    #     """获取当前机器人状态并转换为JSON（异步版，内置状态标准化逻辑）"""
    #     if not self.connected:
    #         raise RuntimeError("机器人未连接，无法获取状态")
        
    #     try:
    #         # 主动读取最新状态，确保数据新鲜
    #         latest_status = await self._service.read_snapshot()
    #         if latest_status:
    #             self.robot_status = latest_status  # 更新缓存的状态

    #         # 定义内部函数：将状态对象转换为兼容旧版的结构
    #         def _normalize_robot_status(status_obj):
    #             from types import SimpleNamespace

    #             # 内部辅助函数：创建单个关节的命名空间
    #             def make_joint(i: int):
    #                 j = SimpleNamespace()
    #                 # 状态对象的列表是0索引，关节编号是1-6，需转换
    #                 j.ActualPosition = float(getattr(status_obj, 'JointActualPosition', [0]*6)[i-1]) if (i-1 < len(getattr(status_obj, 'JointActualPosition', []))) else 0.0
    #                 j.ActualVelocity = float(getattr(status_obj, 'JointActualVelocity', [0]*6)[i-1]) if (i-1 < len(getattr(status_obj, 'JointActualVelocity', []))) else 0.0
    #                 j.ActualAcceleration = float(getattr(status_obj, 'JointActualAcceleration', [0]*6)[i-1]) if (i-1 < len(getattr(status_obj, 'JointActualAcceleration', []))) else 0.0
    #                 j.ActualCurrent = float(getattr(status_obj, 'JointActualCurrent', [0]*6)[i-1]) if (i-1 < len(getattr(status_obj, 'JointActualCurrent', []))) else 0.0
    #                 j.ActualTorque = float(getattr(status_obj, 'JointActualTorque', [0]*6)[i-1]) if (i-1 < len(getattr(status_obj, 'JointActualTorque', []))) else 0.0
    #                 return j

    #             # 构建兼容旧版的状态结构
    #             legacy = SimpleNamespace()
    #             # 关节信息（J1-J6）
    #             joints_ns = SimpleNamespace()
    #             for idx in range(1, 7):
    #                 setattr(joints_ns, f'J{idx}', make_joint(idx))
    #             legacy.Joints = joints_ns
    #             # 管理信息（运动状态、使能状态等）
    #             manage = SimpleNamespace()
    #             manage.Moving = bool(getattr(status_obj, 'Moving', False))
    #             manage.Enabled = bool(getattr(status_obj, 'PowerOn', False))
    #             manage.Mode = int(getattr(status_obj, 'Mode', 0)) if getattr(status_obj, 'Mode', None) is not None else 0
    #             manage.Error = bool(getattr(status_obj, 'Error', False))
    #             legacy.Manage = manage
    #             # TCP位姿信息
    #             tcp = SimpleNamespace()
    #             tp = getattr(status_obj, 'TcpPose', [0.0]*6)
    #             try:
    #                 tcp.X = float(tp[0])
    #                 tcp.Y = float(tp[1])
    #                 tcp.Z = float(tp[2])
    #                 tcp.Roll = float(tp[3])
    #                 tcp.Pitch = float(tp[4])
    #                 tcp.Yaw = float(tp[5])
    #             except Exception:
    #                 tcp.X = tcp.Y = tcp.Z = tcp.Roll = tcp.Pitch = tcp.Yaw = 0.0
    #             legacy.TcpPose = tcp
    #             # 提供to_dict方法用于序列化
    #             def to_dict():
    #                 return {
    #                     'Joints': {f'J{i}': vars(getattr(legacy.Joints, f'J{i}')) for i in range(1,7)},
    #                     'Manage': vars(legacy.Manage),
    #                     'TcpPose': vars(legacy.TcpPose)
    #                 }
    #             legacy.to_dict = to_dict
    #             return legacy

    #         # 处理状态字典的生成（优先使用自带to_dict，否则标准化）
    #         to_dict_fn = getattr(self.robot_status, 'to_dict', None)
    #         if callable(to_dict_fn):
    #             status_dict = to_dict_fn()
    #         else:
    #             legacy = _normalize_robot_status(self.robot_status)
    #             status_dict = legacy.to_dict()

    #         # 转换为JSON并返回
    #         transformed = transform_json(status_dict)
    #         return json.dumps(transformed, indent=4)
        
    #     except Exception as e:
    #         LOGGER.error(f"获取机器人状态失败：{e}", exc_info=True)
    #         raise


    # 获取机器人状态，并转换为JSON
    async def getRobotStatus(self) -> str:
        if not self.connected:
            raise RuntimeError("机器人未连接，无法获取状态")
        
        try:
            latest_status = await self._service.read_snapshot()
            if not latest_status:
                raise ValueError("读取状态为空")
            self.robot_status = latest_status
            
            # 显式断言非None，告诉Pylance self.robot_status一定有值
            assert self.robot_status is not None, "robot_status不能为空"
            
            # 若to_dict是动态添加的，用# type: ignore忽略类型检查
            status_dict = self.robot_status.to_dict()  # type: ignore
            
            transformed = transform_json(status_dict)
            return json.dumps(transformed, indent=4)
        
        except Exception as e:
            LOGGER.error(f"获取机器人状态失败：{e}", exc_info=True)
            raise

    # 关节正向连续点动（速度模式）
    async def JogForward(self, index: int, value: bool) -> None:
        """关节正向连续点动（速度模式）"""
        if not self.connected:
            raise RuntimeError("机器人未连接，无法执行正向点动")
        
        try:
            # 1. 设置控制模式为关节点动模式
            await self.SetControlMode(ControlMode.JointJog)
            # 2. 设置点动模式为速度模式（连续点动）
            await self._service.write_uint('Parameters.Movement_Mode', int(JogMode.VelocityJog))
            # 3. 发送正向点动指令（value为True启动，False停止）
            key = f'Instructions.Joint_Inch_Forward[{index}]'
            await self._service.write_bool(key, value)
            await asyncio.sleep(0.1)  # 等待指令生效
            LOGGER.info(f"关节 {index} 正向连续点动已{'启动' if value else '停止'}")
        except Exception as e:
            LOGGER.error(f"关节 {index} 正向连续点动失败：{e}")
            raise

    # 关节正向增量点动（距离模式）        
    async def InchForward(self, index: int, value: bool) -> None:
        """关节正向增量点动（距离模式）"""
        if not self.connected:
            raise RuntimeError("机器人未连接，无法执行正向增量点动")
        
        try:
            # 1. 设置控制模式为关节点动模式
            await self.SetControlMode(ControlMode.JointJog)
            # 2. 设置点动模式为距离模式（增量点动）
            await self._service.write_uint('Parameters.Movement_Mode', int(JogMode.DistanceJog))
            # 3. 发送正向增量点动指令（value为True触发，False复位）
            key = f'Instructions.Joint_Inch_Forward[{index}]'
            await self._service.write_bool(key, value)
            await asyncio.sleep(0.1)
            LOGGER.info(f"关节 {index} 正向增量点动已{'触发' if value else '复位'}")
        except Exception as e:
            LOGGER.error(f"关节 {index} 正向增量点动失败：{e}")
            raise

    # 关节反向连续点动（速度模式）
    async def JogBackward(self, index: int, value: bool) -> None:
        """关节反向连续点动（速度模式）"""
        if not self.connected:
            raise RuntimeError("机器人未连接，无法执行反向点动")
        
        try:
            # 1. 设置控制模式为关节点动模式
            await self.SetControlMode(ControlMode.JointJog)
            # 2. 设置点动模式为速度模式（连续点动）
            await self._service.write_uint('Parameters.Movement_Mode', int(JogMode.VelocityJog))
            # 3. 发送反向点动指令
            key = f'Instructions.Joint_Inch_Backward[{index}]'
            await self._service.write_bool(key, value)
            await asyncio.sleep(0.1)
            LOGGER.info(f"关节 {index} 反向连续点动已{'启动' if value else '停止'}")
        except Exception as e:
            LOGGER.error(f"关节 {index} 反向连续点动失败：{e}")
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
            await self._service.write_uint('Parameters.Movement_Mode', int(JogMode.DistanceJog))
            # 3. 发送反向增量点动指令
            key = f'Instructions.Joint_Inch_Backward[{index}]'
            await self._service.write_bool(key, value)
            await asyncio.sleep(0.1)
            LOGGER.info(f"关节 {index} 反向增量点动已{'触发' if value else '复位'}")
        except Exception as e:
            LOGGER.error(f"关节 {index} 反向增量点动失败：{e}")
            raise

    # 关节点动统一接口
    async def jointJog(self, axis: str, direction: int, type: str, isStart: bool) -> None:
        """
        关节点动控制（统一接口）
        :param axis: 关节名称（如"J1"、"J2"）
        :param direction: 方向（1:正向，-1:反向）
        :param type: 模式（"speed":速度模式，"distance":距离模式）
        :param isStart: 是否启动（True:启动，False:停止）
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
            await self._service.write_uint('Jog.Mode', int(jog_mode))
            
            # 3. 设置控制模式为关节点动
            await self.SetControlMode(ControlMode.JointJog)
            
            # 4. 根据方向发送点动指令
            if direction == 1:
                await self.JogForward(axisIndex, isStart)  # 复用之前的异步方法
            elif direction == -1:
                await self.JogBackward(axisIndex, isStart)  # 复用之前的异步方法
            else:
                raise ValueError(f"无效的方向：{direction}（需为1或-1）")
            
            LOGGER.info(f"关节 {axis} 点动（{type}模式）已{'启动' if isStart else '停止'}，方向：{direction}")
        except Exception as e:
            LOGGER.error(f"关节点动失败：{e}")
            raise

    # 关节空间运动
    async def MoveJ(self, joint_positions: List[float]) -> None:
        """
        关节空间运动（非阻塞）
        :param joint_positions: 目标关节位置列表（[J1, J2, J3, J4, J5, J6]）
        """
        if not self.connected:
            raise RuntimeError("机器人未连接，无法执行MoveJ")
        if len(joint_positions) != 6:
            raise ValueError("关节位置列表必须包含6个元素（J1-J6）")
        
        try:
            # 1. 写入目标关节位置
            for i in range(6):
                key = f'Parameters.Joint_Target_Position[{i+1}]'  # TODO 索引测试
                await self._service.write_real(key, joint_positions[i])
            
            # 2. 设置控制模式为关节运动模式
            await self.SetControlMode(ControlMode.MoveJoint)
            
            # 3. 触发关节运动执行 
            # TODO 核对key
            await self._service.write_bool('Instructions.TCP_Move_Joint_Execute', True)
            await asyncio.sleep(0.1)  # 等待触发生效
            # （可选）复位触发信号（根据设备需求）
            await self._service.write_bool('Instructions.TCP_Move_Joint_Execute', False)

            LOGGER.info("MoveJ 指令已发送")
        except Exception as e:
            LOGGER.error(f"MoveJ 执行失败：{e}")
            raise

    # 关节空间运动
    async def MoveJWait(self, joint_positions: List[float]) -> None:
        """
        关节空间运动（阻塞，等待完成）
        :param joint_positions: 目标关节位置列表（[J1, J2, J3, J4, J5, J6]）
        """
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
                LOGGER.info(f"MoveJ 完成，到达目标位置（误差：{error}）")
                print("已到达目标位置")
            else:
                LOGGER.warning(f"MoveJ 误差过大：{error}")
                raise Exception(f"未到达目标位置（误差：{error}）")
                
        except Exception as e:
            LOGGER.error(f"MoveJWait 执行失败：{e}")
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
                key = f'Parameters.TCP_Target_Pose[{i+1}]'  # TODO 核对索引
                await self._service.write_real(key, pose[i])
            
            # 2. 设置控制模式为线性运动模式
            await self.SetControlMode(ControlMode.MoveLinear)
            
            # 3. 触发线性运动执行
            await self._service.write_bool('Instructions.TCP_Move_Path_Joint_Execute', True)
            await asyncio.sleep(0.1)  # 等待触发生效
            # （可选）复位触发信号
            await self._service.write_bool('Instructions.TCP_Move_Path_Joint_Execute', False)
            
            LOGGER.info("MoveL 指令已发送")
        except Exception as e:
            LOGGER.error(f"MoveL 执行失败：{e}")
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
                LOGGER.info(f"MoveL 完成，到达目标位置（位置误差：{position_error}，姿态误差：{rotation_error}）")
                print("已到达目标位置")
            else:
                error_msg = []
                if position_error >= 1:
                    error_msg.append(f"位置误差过大：{position_error}")
                if rotation_error >= 1:
                    error_msg.append(f"姿态误差过大：{rotation_error}")
                LOGGER.warning("; ".join(error_msg))
                raise Exception("; ".join(error_msg))
                
        except Exception as e:
            LOGGER.error(f"MoveLWait 执行失败：{e}")
            raise
       
    # TCP相对运动
    async def TcpMoveRel(self, tcp_move_rel_type: Union[int, str], tcp_positions: List[float]) -> None:
        """
        TCP空间相对运动（非阻塞）
        :param tcp_move_rel_type: 相对运动类型（如坐标系类型，int或str）
        :param tcp_positions: 相对位移量列表（[dx, dy, dz, dRoll, dPitch, dYaw]）
        """
        if not self.connected:
            raise RuntimeError("机器人未连接，无法执行TCP相对运动")
        if len(tcp_positions) != 6:
            raise ValueError("相对位移列表必须包含6个元素（dx, dy, dz, dRoll, dPitch, dYaw）")
        
        try:
            # 1. 设置TCP相对运动类型 
            # TODO 核对key
            await self._service.write_uint('TCPMoveRel.Type', int(tcp_move_rel_type))
            
            # 2. 写入相对位移量
            for i in range(6):
                key = f'Parameters.TCP_Relative_Distance[{i+1}]'  # TODO 核对索引
                await self._service.write_real(key, tcp_positions[i])
            
            # 3. 设置控制模式为TCP相对运动模式
            await self.SetControlMode(ControlMode.TCPMoveRel)
            
            # 4. 触发TCP相对运动执行
            await self._service.write_bool('Instructions.TCP_Move_Rel_Execute', True)
            await asyncio.sleep(0.1)  # 等待触发生效
            await self._service.write_bool('Instructions.TCP_Move_Rel_Execute', False)  # 复位触发信号
            
            LOGGER.info("TCP相对运动指令已发送")
        except Exception as e:
            LOGGER.error(f"TCP相对运动执行失败：{e}")
            raise

    # TCP空间相对运动
    async def TcpMoveRelWait(self, tcp_move_rel_type: Union[int, str], tcp_positions: List[float]) -> None:
        """
        TCP空间相对运动（阻塞，等待完成）
        :param tcp_move_rel_type: 相对运动类型（0/Rotation 或 1/Translation，支持int或str）
        :param tcp_positions: 相对位移量列表（[dx, dy, dz, dRoll, dPitch, dYaw]）
        """
        if not self.connected:
            raise RuntimeError("机器人未连接，无法执行TCP相对运动（等待完成）")
        
        try:
            # 关键：将 int | str 转换为 TcpMoveRelType 枚举
            rel_type: TcpMoveRelType  # 显式标注类型，帮助Pylance识别
            if isinstance(tcp_move_rel_type, int):
                # 整数转换：直接匹配枚举的value（0或1）
                rel_type = TcpMoveRelType(tcp_move_rel_type)
            elif isinstance(tcp_move_rel_type, str):
                # 字符串转换：先尝试匹配枚举名称（如"Rotation"），再尝试转换为整数
                try:
                    # 尝试通过名称匹配（不区分大小写，如"rotation"也能匹配）
                    rel_type = TcpMoveRelType[tcp_move_rel_type.capitalize()]
                except KeyError:
                    # 名称匹配失败，尝试将字符串转为整数（如"0" → 0）
                    try:
                        rel_type = TcpMoveRelType(int(tcp_move_rel_type))
                    except (ValueError, TypeError):
                        raise ValueError(f"无效的字符串类型：{tcp_move_rel_type}（需为'Rotation'/'Translation'或'0'/'1'）")
            else:
                raise TypeError(f"不支持的类型：{type(tcp_move_rel_type)}（需为int或str）")

            # 1. 获取初始位姿
            start_status = await self._service.read_snapshot()
            if not start_status:
                raise RuntimeError("无法获取初始位姿，读取状态失败")
            
            start_pose = [
                start_status.TcpPose[0], start_status.TcpPose[1], start_status.TcpPose[2],
                start_status.TcpPose[3], start_status.TcpPose[4], start_status.TcpPose[5]
            ]
            
            # 2. 计算目标位姿（传入转换后的 TcpMoveRelType 枚举）
            T0 = Utils.pose_to_T(start_pose)
            target_pose = Utils.Calculate_Tcp_targetPose(rel_type, tcp_positions, T0, start_pose) 
            
            # 3. 发送相对运动指令（同步修改 TcpMoveRel 方法，确保类型一致）
            await self.TcpMoveRel(rel_type, tcp_positions)  # 传入枚举，内部再转整数写入寄存器
            
            # 4. 等待运动完成（逻辑不变）
            while True:
                current_status = await self._service.read_snapshot()
                if not current_status:
                    raise RuntimeError("读取运动状态失败")
                if not current_status.Moving:
                    break
                await asyncio.sleep(0.1)
            
            # 5. 检查误差（逻辑不变）
            actual_pose = [
                current_status.TcpPose[0], current_status.TcpPose[1], current_status.TcpPose[2],
                current_status.TcpPose[3], current_status.TcpPose[4], current_status.TcpPose[5]
            ]
            position_error, rotation_error = Utils.caculate_cartesian_error(target_pose, actual_pose)
            
            if position_error < 0.5 and rotation_error < 0.5:
                LOGGER.info(f"TCP相对运动完成（位置误差：{position_error}，姿态误差：{rotation_error}）")
                print("已到达目标位置")
            else:
                error_msg = []
                if position_error >= 0.5:
                    error_msg.append(f"位置误差过大：{position_error}")
                if rotation_error >= 0.5:
                    error_msg.append(f"姿态误差过大：{rotation_error}")
                LOGGER.warning("; ".join(error_msg))
                raise Exception("; ".join(error_msg))
                
        except Exception as e:
            LOGGER.error(f"TCP相对运动（等待完成）执行失败：{e}")
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
            key = f'Parameters.Joint_Target_Position[{joint_index}]'
            await self._service.write_real(key, target_position)
            
            # 2. 设置控制模式为关节绝对运动模式
            await self.SetControlMode(ControlMode.JointMoveAbs)

            # 3. 触发单关节绝对运动执行
            execute_key = f'Instructions.Joint_MoveABS_Execute[{joint_index}]'
            await self._service.write_bool(execute_key, True)
            await asyncio.sleep(0.1)  # 等待触发生效
            await self._service.write_bool(execute_key, False)  # 复位触发信号
            
            LOGGER.info(f"关节 {joint_index} 绝对运动指令已发送（目标位置：{target_position}）")
        except Exception as e:
            LOGGER.error(f"关节 {joint_index} 绝对运动执行失败：{e}")
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
                LOGGER.info(f"关节 {joint_index} 绝对运动完成，到达目标位置（误差：{error}）")
                print("已到达目标位置")
            else:
                LOGGER.warning(f"关节 {joint_index} 绝对运动误差过大：{error}")
                raise Exception(f"关节 {joint_index} 未到达目标位置（误差：{error}）")
                
        except Exception as e:
            LOGGER.error(f"关节 {joint_index} 绝对运动（等待完成）执行失败：{e}")
            raise

    # TCP速度控制
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
            # TODO 
            # # 假设地址簿中TCP速度控制的key为'TCP.Velocity[i]'（i=0-5）
            # for i in range(6):
            #     key = f'TCP.Velocity[{i}]'
            #     await self._service.write_real(key, velocity_vector[i])
            
            
            pass
            
            LOGGER.info("TCP速度向量已设置")
        except Exception as e:
            LOGGER.error(f"TCP速度控制执行失败：{e}")
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
            stop_key = f'Instructions.Joint_Stop_Execute[{joint_index}]'
            await self._service.write_bool(stop_key, True)
            await asyncio.sleep(0.1)  
            await self._service.write_bool(stop_key, False)  
            
            LOGGER.info(f"关节 {joint_index} 绝对运动已停止")
        except Exception as e:
            LOGGER.error(f"停止关节 {joint_index} 绝对运动失败：{e}")
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


    # 气锁控制
    async def AirLockOn(self) -> None:
        """气锁开启"""
        await self._set_air_lock(AirLock.On)

    async def AirLockOff(self) -> None:
        """气锁关闭"""
        await self._set_air_lock(AirLock.Off)

    async def AirLockHold(self) -> None:
        """气锁保持"""
        await self._set_air_lock(AirLock.Hold)

    async def _set_air_lock(self, mode: AirLock) -> None:
        """气锁控制底层实现（私有辅助方法）"""
        if not self.connected:
            raise RuntimeError("机器人未连接，无法控制气锁")
        
        try:
            # TODO 气锁的控制量 
            await self._service.write_uint('AirLock.Control', int(mode))
            await asyncio.sleep(0.1)
            LOGGER.info(f"气锁已设置为：{mode.name}")
        except Exception as e:
            LOGGER.error(f"气锁控制失败（模式：{mode.name}）：{e}")
            raise


    # 机器人全局控制
    async def RobotStop(self) -> None:
        """停止机器人所有运动"""
        if not self.connected:
            raise RuntimeError("机器人未连接，无法执行停止操作")
        
        try:
            await self._service.write_bool('Instructions.Stop_Execute', True)
            await asyncio.sleep(0.1)
            await self._service.write_bool('Instructions.Stop_Execute', False)
            LOGGER.info("机器人已停止所有运动")
        except Exception as e:
            LOGGER.error(f"机器人停止失败：{e}")
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


    # 吸盘/电磁铁控制（异步）
    async def suckerOn(self) -> None:
        """吸盘开启（数字输出0置位）"""
        await self.setDigitalOutput(0, True)  # 复用异步的setDigitalOutput


    async def suckerOff(self) -> None:
        """吸盘关闭（数字输出0复位）"""
        await self.setDigitalOutput(0, False)


    async def ElectromagnetOn(self) -> None:
        """电磁铁开启（数字输出1置位）"""
        await self.setDigitalOutput(1, True)


    async def ElectromagnetOff(self) -> None:
        """电磁铁关闭（数字输出1复位）"""
        await self.setDigitalOutput(1, False)


    # TODO 夹爪控制（根据夹爪接口调整，若同步则保持调用，若异步则用await）
    def GripperOpen(self) -> None:
        """夹爪打开（假设夹爪接口为同步，保持原调用）"""
        self.gripper.grip_open_full_no_param()
        LOGGER.info("夹爪已打开")


    def GripperClose(self) -> None:
        """夹爪关闭"""
        self.gripper.grip_close_full_no_param()
        LOGGER.info("夹爪已关闭")


    def GripperRotateClockwise(self) -> None:
        """夹爪顺时针旋转"""
        self.gripper.rotate_relative(504, 0xFF, 0xFF)
        LOGGER.info("夹爪已顺时针旋转")


    def GripperRotateCounterClockwise(self) -> None:
        """夹爪逆时针旋转"""
        self.gripper.rotate_relative(-900, 0xFF, 0xFF)
        LOGGER.info("夹爪已逆时针旋转")

    # -------------------------------------
    # 辅助方法
    # -------------------------------------
    # 将 tcp_pose 转换为列表
    # def _tcp_pose_to_list(self, tcp_pose) -> list:
    #     """返回一个 [X,Y,Z,Roll,Pitch,Yaw] 列表，tcp_pose 可能是一个浮点数列表
    #     （由 Communication.ModBusService.RobotStatus 生成）或具有属性
    #     X/Y/Z/Roll/Pitch/Yaw 的对象（旧版形状）。"""
    #     # list/tuple from ModBusService
    #     if isinstance(tcp_pose, (list, tuple)):
    #         # ensure length 6
    #         vals = list(tcp_pose)[:6] + [0.0] * max(0, 6 - len(tcp_pose))
    #         return [float(v) for v in vals]
    #     # object with attributes
    #     try:
    #         return [
    #             float(getattr(tcp_pose, 'X', 0.0)),
    #             float(getattr(tcp_pose, 'Y', 0.0)),
    #             float(getattr(tcp_pose, 'Z', 0.0)),
    #             float(getattr(tcp_pose, 'Roll', 0.0)),
    #             float(getattr(tcp_pose, 'Pitch', 0.0)),
    #             float(getattr(tcp_pose, 'Yaw', 0.0)),
    #         ]
    #     except Exception:
    #         return [0.0] * 6
