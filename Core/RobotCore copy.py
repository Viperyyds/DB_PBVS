import time
from threading import Timer
from Core.Basic import *
from Core.ModbusInterface import *
from Util import *
import json
from pprint import pprint
from Util import *
from Core.gripper import Gripper

class RobotCore:
    def __init__(self,target_ip:str):
        self.communicator = Communicator(target_ip, 502) 
        self.communicator.Connect()
        self.robot_parameter = RobotParameters()
        self.robot_status = RobotStatus()
        self.gripper = Gripper("COM4")
        self.timer = Timer(0.05, self.ReadRobotStatus)
    

        time.sleep(0.5)
        if self.communicator.robot_connected:
            print("连接成功！")
            self.timer.start()
            self.communicator.RobotEnable()
            self.communicator.RobotReset()
            self.communicator.SetControlMode(ControlMode.Idel)
            # 设置运动参数
            self.communicator.RobotSetParameters(self.robot_parameter)
            print("初始化成功！")
        else:
            print("连接失败！")

    def setDigitalOutput(self, index: int, value: bool):
        time.sleep(0.5)
        self.communicator.SetDigitalOutput(index, value)
        time.sleep(0.5)

    def ReadRobotStatus(self):
        self.robot_status = self.communicator.GetRobotState()
        # print_object(self.robot_status)
        # self.timer = Timer(0.05, self.ReadRobotStatus)
        # self.timer.start()
        
    def getRobotStatus(self):
        self.robot_status = self.communicator.GetRobotState()
        # print_object(self.robot_status)

        robot_status_json = json.dumps(transform_json(self.robot_status.to_dict()), indent=4)
        # print(robot_status_json)
        
        return robot_status_json

    def JogForward(self,index:int,value:bool):
        self.communicator.SetControlMode(ControlMode.JointJog)
        self.communicator.SetJogMod(JogMode.VelocityJog)
        self.communicator.JogForward(index,value)


    def InchForward(self,index:int, value:bool):
        self.communicator.SetControlMode(ControlMode.JointJog)
        self.communicator.SetJogMod(JogMode.DistanceJog)
        self.communicator.JogForward(index, value)

    # def InchForwardWait(self,index:int,value:bool):
    #     self.communicator.SetJogMod(JogMode.DistanceJog)
    #     self.communicator.JogForward(index,value)
    #     while True:
    #         if self.robot_status.Manage.Moving:
    #             sleep(0.05)
    #         else:
    #             break


    def JogBackward(self,index:int,value:bool):
        self.communicator.SetControlMode(ControlMode.JointJog)
        self.communicator.SetJogMod(JogMode.VelocityJog)
        self.communicator.JogBackward(index,value)

    def InchBackward(self,index:int,value:bool):
        self.communicator.SetControlMode(ControlMode.JointJog)
        self.communicator.SetJogMod(JogMode.DistanceJog)
        self.communicator.JogBackward(index, value)

    # def JogStop(self):
    #     self.communicator.JogStop()

    def jointJog(self, axis:str, direction:int, type:str, isStart:bool):
        if type == "speed":
            self.communicator.SetJogMod(JogMode.VelocityJog)
        elif type == "distance":
            self.communicator.SetJogMod(JogMode.DistanceJog)
        
        axisIndex = get_joint_index(axis)
        if direction == 1:
            self.communicator.JogForward(axisIndex, isStart)
        elif direction == -1:
            self.communicator.JogBackward(axisIndex, isStart)

    # def JogBackwardWait(self,index:int,value:bool,mode:JogMode):
    #     self.communicator.SetJogMod(mode)
    #     self.communicator.JogBackward(index,value)
    #     while True:
    #         if self.robot_status.Manage.Moving:
    #             sleep(0.05)
    #         else:
    #             break

    def MoveJ(self,JointPositions):
        self.communicator.SetJointsTargetPosition(JointPositions)
        self.communicator.SetControlMode(ControlMode.MoveJoint)
        self.communicator.ExecuteMoveJoint()

    def MoveJWait(self,JointPositions):
        self.communicator.SetJointsTargetPosition(JointPositions)
        self.communicator.SetControlMode(ControlMode.MoveJoint)
        self.communicator.ExecuteMoveJoint()
        while True:
            sleep(0.1)
            self.robot_status = self.communicator.GetRobotState()
            if self.robot_status.Manage.Moving:
                pass
                # print("正在运动")
            else:
                TargetJointPositions = [self.robot_status.Joints.J1.ActualPosition,
                                 self.robot_status.Joints.J2.ActualPosition,
                                 self.robot_status.Joints.J3.ActualPosition,
                                 self.robot_status.Joints.J4.ActualPosition,
                                 self.robot_status.Joints.J5.ActualPosition,
                                 self.robot_status.Joints.J6.ActualPosition]
                error = Utils.caculate_joints_error(JointPositions, TargetJointPositions)
                if error < 0.1:
                    print("已到达目标位置")
                else:
                    print("综合误差过大：", error)
                    raise Exception("未到达目标位置")
                break


    def MoveL(self,Pose):
        self.communicator.SetCartesianTargetPosition(Pose)
        self.communicator.SetControlMode(ControlMode.MoveLinear)
        self.communicator.ExecuteMoveLinear()

    def MoveLWait(self,Pose):
        self.communicator.SetCartesianTargetPosition(Pose)
        self.communicator.SetControlMode(ControlMode.MoveLinear)
        self.communicator.ExecuteMoveLinear()
        sleep(1)
        while True:
            self.robot_status = self.communicator.GetRobotState()
            if self.robot_status.Manage.Moving:
                pass
                # print("正在运动")
            else:
                ActualPose = [self.robot_status.TcpPose.X,
                              self.robot_status.TcpPose.Y,
                              self.robot_status.TcpPose.Z,
                              self.robot_status.TcpPose.Roll,
                              self.robot_status.TcpPose.Pitch,
                              self.robot_status.TcpPose.Yaw]
                position_error,rotation_error = Utils.caculate_cartesian_error(Pose, ActualPose)

                if position_error < 1 and rotation_error < 1:
                    print("已到达目标位置")
                    break
                elif position_error > 1 and rotation_error < 1:
                    print("位置误差过大：", position_error)
                elif position_error < 1 and rotation_error > 1:
                    print("姿态误差过大：", rotation_error)
                else:
                    print("位置误差过大：", position_error)
                    print("姿态误差过大：", rotation_error)
            sleep(0.5)

    def TcpMoveRel(self,TcpMoveRelType,TcpPositions):
        self.communicator.SetTcpMoveType(TcpMoveRelType)
        self.communicator.SetTcpMoveRelPosition(TcpPositions)
        self.communicator.SetControlMode(ControlMode.TCPMoveRel)
        self.communicator.ExecuteTCPMoveRel()

    def TcpMoveRelWait(self,TcpMoveRelType,TcpPositions):
        self.communicator.SetTcpMoveType(TcpMoveRelType)
        self.communicator.SetTcpMoveRelPosition(TcpPositions)
        self.communicator.SetControlMode(ControlMode.TCPMoveRel)
        # 获取机器人开始运动的初始位姿
        self.robot_status = self.communicator.GetRobotState()
        start_pose = [self.robot_status.TcpPose.X,self.robot_status.TcpPose.Y,self.robot_status.TcpPose.Z,self.robot_status.TcpPose.Roll,self.robot_status.TcpPose.Pitch,self.robot_status.TcpPose.Yaw]
        T0 = Utils.pose_to_T(start_pose)
        target_pose = Utils.Calculate_Tcp_targetPose(TcpMoveRelType,TcpPositions,T0,start_pose)
        self.communicator.ExecuteTCPMoveRel()
        while True:
            # todo 需完善
            self.robot_status = self.communicator.GetRobotState()
            if self.robot_status.Manage.Moving:
                print("正在运动")
                sleep(0.1)
            else:
                ActualPose = [self.robot_status.TcpPose.X,
                              self.robot_status.TcpPose.Y,
                              self.robot_status.TcpPose.Z,
                              self.robot_status.TcpPose.Roll,
                              self.robot_status.TcpPose.Pitch,
                              self.robot_status.TcpPose.Yaw]
                position_error, rotation_error = Utils.caculate_cartesian_error(target_pose, ActualPose)

                if position_error < 0.5 and rotation_error < 0.5:
                    print("已到达目标位置")
                elif position_error > 0.5 and rotation_error < 0.5:
                    print("位置误差过大：", position_error)
                    raise Exception("未到达目标位置")
                elif position_error < 0.5 and rotation_error > 0.5:
                    print("姿态误差过大：", rotation_error)
                    raise Exception("未到达目标位置")
                else:
                    print("位置误差过大：", position_error)
                    print("姿态误差过大：", rotation_error)
                    raise Exception("未到达目标位置")
                break

    
    def MoveAbs(self,joint_index:int,targetPosition:float):
        self.communicator.SetJointMoveABSPosition(joint_index, targetPosition)
        self.communicator.SetControlMode(ControlMode.JointMoveAbs)
        self.communicator.ExecuteJointMoveABS(joint_index)
        
    def MoveABSWait(self,joint_index:int,targetPosition:float):
        self.communicator.SetJointMoveABSPosition(joint_index, targetPosition)
        self.communicator.SetControlMode(ControlMode.JointMoveAbs)
        self.communicator.ExecuteJointMoveABS(joint_index)
        while True:
            # todo 需完善
            self.robot_status = self.communicator.GetRobotState()
            if self.robot_status.Manage.Moving:
                print("正在运动")
                sleep(0.1)
            else:
                CurrentJointPositions = [self.robot_status.Joints.J1.ActualPosition,
                                        self.robot_status.Joints.J2.ActualPosition,
                                        self.robot_status.Joints.J3.ActualPosition,
                                        self.robot_status.Joints.J4.ActualPosition,
                                        self.robot_status.Joints.J5.ActualPosition,
                                        self.robot_status.Joints.J6.ActualPosition]

                error = abs(targetPosition - CurrentJointPositions[joint_index-1])
                if error < 0.1:
                    print("已到达目标位置")
                else:
                    print("综合误差过大",error)
                    raise Exception("未到达目标位置")
                break

    def Move_tcp_velocity(self,velocity_vector):
        pass

    def MoveAbsStop(self,joint_index:int):
        self.communicator.StopJointMoveAbs(joint_index)
        
    def JointMoveAbs(self, joint_name:str, isStop: bool, targetPosition:float = None):
        joint_index = get_joint_index(joint_name)
        if not isStop:
            self.MoveAbs(joint_index, targetPosition)
        else:
            self.MoveAbsStop(joint_index)

    def AirLockOn(self):
        self.communicator.AirLockControl(AirLock.On)

    def AirLockOff(self):
        self.communicator.AirLockControl(AirLock.Off)

    def AirLockHold(self):
        self.communicator.AirLockControl(AirLock.Hold)
    
    def RobotStop(self):
        self.communicator.RobotStop()
        
    def RobotReset(self):
        self.communicator.RobotReset()
    
    def RobotEnable(self):
        self.communicator.RobotEnable()
        
    def RobotDisable(self):
        self.communicator.RobotDisable()
        
    def activateJointMoveAbs(self):
        self.communicator.SetControlMode(ControlMode.JointMoveAbs)

    def activateMoveLinear(self):
        self.communicator.SetControlMode(ControlMode.MoveLinear)

    def activateMoveJoint(self):
        self.communicator.SetControlMode(ControlMode.MoveJoint)

    def activateJointJog(self):
        self.communicator.SetControlMode(ControlMode.JointJog)

    def suckerOn(self):
        self.setDigitalOutput(0, True)

    def suckerOff(self):
        self.setDigitalOutput(0, False)

    def ElectromagnetOn(self):
        self.setDigitalOutput(1, True)

    def ElectromagnetOff(self):
        self.setDigitalOutput(1, False)

    def GripperOpen(self):
        self.gripper.grip_open_full_no_param()

    def GripperClose(self):
        self.gripper.grip_close_full_no_param()

    def GripperRotateClockwise(self):
        self.gripper.rotate_relative(504,0xFF,0xFF)

    def GripperRotateCounterClockwise(self):
        self.gripper.rotate_relative(-900,0xFF,0xFF)

    # def activateTcpMoveRel(self):
    #     self.communicator.SetControlMode(ControlMode.TCPMoveRel)




