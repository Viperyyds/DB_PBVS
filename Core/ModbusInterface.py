import threading
from time import sleep

from pymodbus.client.tcp import ModbusTcpClient
from pymodbus.constants import Endian
from pymodbus.payload import BinaryPayloadDecoder, BinaryPayloadBuilder

from Core.Basic import *
from Core.UtilsData import *


class AddressTable:
    # modbus中的线圈
    Power_Enable = 0
    Reset_Execute = 1
    Stop_Execute = 2
    Parameter_Write_Enable = 3

    Jog_Forward_1 = 8
    Jog_Forward_2 = 9
    Jog_Forward_3 = 10
    Jog_Forward_4 = 11
    Jog_Forward_5 = 12
    Jog_Forward_6 = 13

    Move_Velocity_Execute = 16
    Move_Velocity_Abort = 17

    Jog_Backward_1 = 24
    Jog_Backward_2 = 25
    Jog_Backward_3 = 26
    Jog_Backward_4 = 27
    Jog_Backward_5 = 28
    Jog_Backward_6 = 29

    Joint1_MoveABS_Execute = 32
    Joint2_MoveABS_Execute = 33
    Joint3_MoveABS_Execute = 34
    Joint4_MoveABS_Execute = 35
    Joint5_MoveABS_Execute = 36
    Joint6_MoveABS_Execute = 37

    TCP_Move_Rel_Execute = 40
    TCP_Move_Rel_Abort = 40

    Move_Joint_Execute = 48
    Move_Joint_Abort = 49
    Move_Linear_Execute = 50
    Move_Linear_Abort = 51
    Move_Path_Joint_Execute = 52
    Move_Path_Joint_Abort = 53
    Move_Path_Linear_Execute = 54
    Move_Path_Linear_Abort = 55

    TCP_Jog_along_X_Positive = 56
    TCP_Jog_along_X_Negative = 57
    TCP_Jog_along_Y_Positive = 58
    TCP_Jog_along_Y_Negative = 59
    TCP_Jog_along_Z_Positive = 60
    TCP_Jog_along_Z_Negative = 61

    TCP_Jog_around_X_Positive = 64
    TCP_Jog_around_X_Negative = 65
    TCP_Jog_around_Y_Positive = 66
    TCP_Jog_around_Y_Negative = 67
    TCP_Jog_around_Z_Positive = 68
    TCP_Jog_around_Z_Negative = 69

    Joint1_Stop_Execute = 72
    Joint2_Stop_Execute = 73
    Joint3_Stop_Execute = 74
    Joint4_Stop_Execute = 75
    Joint5_Stop_Execute = 76
    Joint6_Stop_Execute = 77

    RobotAdmittanceControl_Execute = 80
    RobotAdmittanceControl_Abort = 81

    # 力传感器参数软标定
    FTSensor_Software_Execute = 89

    DigitalOutPut_0 = 104
    DigitalOutPut_1 = 105
    DigitalOutPut_2 = 106
    DigitalOutPut_3 = 107
    DigitalOutPut_4 = 108
    DigitalOutPut_5 = 109
    DigitalOutPut_6 = 110
    DigitalOutPut_7 = 111

    # modbus中的离散输入
    Initialized = 0
    Enabled = 1
    Moving = 2
    Error = 3

    DigitalInput_0 = 8
    DigitalInput_1 = 9
    DigitalInput_2 = 10
    DigitalInput_3 = 11
    DigitalInput_4 = 12
    DigitalInput_5 = 13
    DigitalInput_6 = 14
    DigitalInput_7 = 15

    # modbus中的输入寄存器
    ErrorID = 0
    Joint1_Status = 1
    Joint2_Status = 2
    Joint3_Status = 3
    Joint4_Status = 4
    Joint5_Status = 5
    Joint6_Status = 6

    Joint1_Actual_Position = 7
    Joint2_Actual_Position = 11
    Joint3_Actual_Position = 15
    Joint4_Actual_Position = 19
    Joint5_Actual_Position = 23
    Joint6_Actual_Position = 27

    Joint1_Actual_Velocity = 31
    Joint2_Actual_Velocity = 35
    Joint3_Actual_Velocity = 39
    Joint4_Actual_Velocity = 43
    Joint5_Actual_Velocity = 47
    Joint6_Actual_Velocity = 51

    Joint1_Actual_Acceleration = 55
    Joint2_Actual_Acceleration = 59
    Joint3_Actual_Acceleration = 63
    Joint4_Actual_Acceleration = 67
    Joint5_Actual_Acceleration = 71
    Joint6_Actual_Acceleration = 75

    Joint1_Actual_Current = 79
    Joint2_Actual_Current = 83
    Joint3_Actual_Current = 87
    Joint4_Actual_Current = 91
    Joint5_Actual_Current = 95
    Joint6_Actual_Current = 99

    Flange_Pose_X = 103
    Flange_Pose_Y = 107
    Flange_Pose_Z = 111
    Flange_Pose_Roll = 115
    Flange_Pose_Pitch = 119
    Flange_Pose_Yaw = 123

    TCP_Pose_X = 127
    TCP_Pose_Y = 131
    TCP_Pose_Z = 135
    TCP_Pose_Roll = 139
    TCP_Pose_Pitch = 143
    TCP_Pose_Yaw = 147

    # 六维力传感器的值
    Fx = 151
    Fy = 155
    Fz = 159
    Tx = 163
    Ty = 167
    Tz = 171

    # modbus中的保持寄存器
    Ctrl_Mode = 0
    AirLock = 1
    TCP_Relative_Type = 2

    DH_D1 = 3
    DH_D3 = 7
    DH_D6 = 11

    Override = 15

    Tool_X = 19
    Tool_Y = 23
    Tool_Z = 27
    Tool_Roll = 31
    Tool_Pitch = 35
    Tool_Yaw = 39

    Jog_Mode = 43

    Jog_IncDistance_1 = 44
    Jog_IncDistance_2 = 48
    Jog_IncDistance_3 = 52
    Jog_IncDistance_4 = 56
    Jog_IncDistance_5 = 60
    Jog_IncDistance_6 = 64

    Jog_Velocity_1 = 68
    Jog_Velocity_2 = 72
    Jog_Velocity_3 = 76
    Jog_Velocity_4 = 80
    Jog_Velocity_5 = 84
    Jog_Velocity_6 = 88

    Joint1_Target_Position = 92
    Joint2_Target_Position = 96
    Joint3_Target_Position = 100
    Joint4_Target_Position = 104
    Joint5_Target_Position = 108
    Joint6_Target_Position = 112

    Joint1_Refference_Velocity = 116
    Joint2_Refference_Velocity = 120
    Joint3_Refference_Velocity = 124
    Joint4_Refference_Velocity = 128
    Joint5_Refference_Velocity = 132
    Joint6_Refference_Velocity = 136

    Joint1_Refference_Acceleration = 140
    Joint2_Refference_Acceleration = 144
    Joint3_Refference_Acceleration = 148
    Joint4_Refference_Acceleration = 152
    Joint5_Refference_Acceleration = 156
    Joint6_Refference_Acceleration = 160

    Joint1_Refference_Jerk = 164
    Joint2_Refference_Jerk = 168
    Joint3_Refference_Jerk = 172
    Joint4_Refference_Jerk = 176
    Joint5_Refference_Jerk = 180
    Joint6_Refference_Jerk = 184

    MoveJ_Refference_Velocity = 188
    MoveJ_Refference_Acceleration = 192
    MoveJ_Refference_Deceleration = 196


    TCP_Target_Position_X = 200
    TCP_Target_Position_Y = 204
    TCP_Target_Position_Z = 208
    TCP_Target_Position_Roll = 212
    TCP_Target_Position_Pitch = 216
    TCP_Target_Position_Yaw = 220

    MoveL_Refference_Linear_Velocity = 224
    MoveL_Refference_Linear_Acceleration = 228
    MoveL_Refference_Linear_Deceleration = 232
    MoveL_Refference_Angular_Velocity = 236
    MoveL_Refference_Angular_Acceleration = 240
    MoveL_Refference_Angular_Deceleration = 244

    TCP_Relative_Distance_along_X = 248
    TCP_Relative_Distance_along_Y = 252
    TCP_Relative_Distance_along_Z = 256
    TCP_Relative_Distance_around_X = 260
    TCP_Relative_Distance_around_Y = 264
    TCP_Relative_Distance_around_Z = 268


class Communicator:
    def __init__(self, ip, port):
        """
        初始化 Communicator 类的实例。
        
        参数:
        ip (str): Modbus TCP 服务器的 IP 地址。
        port (int): Modbus TCP 服务器的端口号。
        """
        # 创建一个 Modbus TCP 客户端实例，连接到指定的 IP 地址和端口
        self.client = ModbusTcpClient(host = ip, port = port)
        # 创建一个线程锁，用于线程安全的操作
        self.lock = threading.Lock()
        # 用于标识机器人是否已连接
        self.robot_connected = False

        self.utils = UtilsData()


    def __set_bool_value_to_address(self, address, value):
        try:
            self.lock.acquire()
            self.client.write_coil(address, value)
            self.lock.release()
            return True
        except Exception as ex:
            print(ex)
            return False

    def __set_double_value_to_address(self, address, value):
        builder = BinaryPayloadBuilder(byteorder=Endian.BIG,wordorder=Endian.LITTLE)
        builder.add_64bit_float(value)
        registers = builder.to_registers()
        try:
            self.lock.acquire()
            self.client.write_registers(address, registers)
            self.lock.release()
            return True
        except Exception as ex:
            print(ex)
            return False
        
    def __set_int_value_to_address(self, address, int_value):
        builder = BinaryPayloadBuilder(byteorder=Endian.BIG, wordorder=Endian.LITTLE)
        builder.add_16bit_int(int_value)
        registers = builder.to_registers()
        try:
            self.lock.acquire()
            self.client.write_registers(address, registers)
            self.lock.release()
            return True
        except Exception as ex:
            print(ex)
            return False
        
    def __get_all_remote_data_registers(self):
        try:
            self.lock.acquire()
            # 离散输入
            discrete_Inputbytes = self.client.read_discrete_inputs(0,8).bits
            # 输入寄存器
            Input_RegisterBytes_1 = self.client.read_input_registers(0,120).registers
            Input_RegisterBytes_2 = self.client.read_input_registers(120,31).registers
            # 保持寄存器
            Holding_RegisterBytes_1 = self.client.read_holding_registers(0,120).registers
            Holding_RegisterBytes_2 = self.client.read_holding_registers(120, 120).registers
            Holding_RegisterBytes_3 = self.client.read_holding_registers(240, 32).registers
            self.lock.release()

            Input_RegisterBytes = Input_RegisterBytes_1 + Input_RegisterBytes_2
            Holding_RegisterBytes = Holding_RegisterBytes_1 + Holding_RegisterBytes_2 + Holding_RegisterBytes_3

            return discrete_Inputbytes, Input_RegisterBytes, Holding_RegisterBytes
        except Exception as ex:
            print(ex)
            return None

    def __get_bool_value_from_address(self, address):
        try:
            self.lock.acquire()
            result = self.client.read_discrete_inputs(address, 1).bits[0]
            self.lock.release()
            return result
        except Exception as ex:
            print(ex)
            return None

    def __get_int_value_from_address(self, address):
        try:
            self.lock.acquire()
            input_registerBytes = self.client.read_input_registers(address, 1).registers
            self.lock.release()
            result = BinaryPayloadDecoder.fromRegisters(input_registerBytes, byteorder=Endian.BIG, wordorder=Endian.LITTLE).decode_16bit_int()
            return result
        except Exception as ex:
            print(ex)
            return None

    def __get_double_value_from_address(self, address):
        try:
            self.lock.acquire()
            input_registerBytes = self.client.read_input_registers(address, 4).registers
            self.lock.release()
            result = BinaryPayloadDecoder.fromRegisters(input_registerBytes, byteorder=Endian.BIG, wordorder=Endian.LITTLE).decode_64bit_float()
            return result
        except Exception as ex:
            print(ex)
            return None
        

    def Connect(self):
        if self.client.connect():
            self.robot_connected = True
        else:
            self.robot_connected = False


    def Disconnect(self):
        if self.robot_connected:
            self.client.close()
            self.robot_connected = False

    def EnableParameterWrite(self):
        sleep(0.1)
        self.__set_bool_value_to_address(AddressTable.Parameter_Write_Enable,True)
        sleep(0.1)

    def DisableParameterWrite(self):
        sleep(0.1)
        self.__set_bool_value_to_address(AddressTable.Parameter_Write_Enable,False)
        sleep(0.1)



    def TestRead(self,address):
        try:
            self.lock.acquire()
            result = self.client.read_holding_registers(address, 1).registers[0]
            self.lock.release()
            return result
        except Exception as ex:
            print(ex)
            return 0
        
    def TestWrite(self,address,value):
        try:
            self.lock.acquire()
            # self.client.write_register(address, value)
            self.client.write_coil(address, value)
            self.lock.release()
            return True
        except Exception as ex:
            print(ex)
            return False
 

    def SetJointsTargetPosition(self,Positions:list):
        self.EnableParameterWrite()
        self.__set_double_value_to_address(AddressTable.Joint1_Target_Position,Positions[0])
        self.__set_double_value_to_address(AddressTable.Joint2_Target_Position,Positions[1])
        self.__set_double_value_to_address(AddressTable.Joint3_Target_Position,Positions[2])
        self.__set_double_value_to_address(AddressTable.Joint4_Target_Position,Positions[3])
        self.__set_double_value_to_address(AddressTable.Joint5_Target_Position,Positions[4])
        self.__set_double_value_to_address(AddressTable.Joint6_Target_Position,Positions[5])
        self.DisableParameterWrite()


    def SetCartesianTargetPosition(self,Pose:list):
        self.EnableParameterWrite()
        self.__set_double_value_to_address(AddressTable.TCP_Target_Position_X,Pose[0])
        self.__set_double_value_to_address(AddressTable.TCP_Target_Position_Y,Pose[1])
        self.__set_double_value_to_address(AddressTable.TCP_Target_Position_Z,Pose[2])
        self.__set_double_value_to_address(AddressTable.TCP_Target_Position_Roll,Pose[3])
        self.__set_double_value_to_address(AddressTable.TCP_Target_Position_Pitch,Pose[4])
        self.__set_double_value_to_address(AddressTable.TCP_Target_Position_Yaw,Pose[5])
        self.DisableParameterWrite()

    def SetTcpMoveRelPosition(self,Positions:list):
        self.EnableParameterWrite()
        self.__set_double_value_to_address(AddressTable.TCP_Relative_Distance_along_X,Positions[0])
        self.__set_double_value_to_address(AddressTable.TCP_Relative_Distance_along_Y,Positions[1])
        self.__set_double_value_to_address(AddressTable.TCP_Relative_Distance_along_Z,Positions[2])
        self.__set_double_value_to_address(AddressTable.TCP_Relative_Distance_around_X,Positions[3])
        self.__set_double_value_to_address(AddressTable.TCP_Relative_Distance_around_Y,Positions[4])
        self.__set_double_value_to_address(AddressTable.TCP_Relative_Distance_around_Z,Positions[5])
        self.DisableParameterWrite()


    def SetJointMoveABSPosition(self,joint_index,position):
        self.EnableParameterWrite()
        if joint_index == 1:
            self.__set_double_value_to_address(AddressTable.Joint1_Target_Position, position)
        elif joint_index == 2:
            self.__set_double_value_to_address(AddressTable.Joint2_Target_Position, position)
        elif joint_index == 3:
            self.__set_double_value_to_address(AddressTable.Joint3_Target_Position, position)
        elif joint_index == 4:
            self.__set_double_value_to_address(AddressTable.Joint4_Target_Position, position)
        elif joint_index == 5:
            self.__set_double_value_to_address(AddressTable.Joint5_Target_Position, position)
        elif joint_index == 6:
            self.__set_double_value_to_address(AddressTable.Joint6_Target_Position, position)
        else:
            print("index error")

        self.DisableParameterWrite()

    def SetTcpVelocityVector(self,velocityVector):
        self.EnableParameterWrite()
        # todo 将下位机的参数暴露出来

    def SetControlMode(self,mode:ControlMode):
        self.__set_int_value_to_address(AddressTable.Ctrl_Mode,mode)
        sleep(0.1)

    def SetTcpMoveType(self,type:TcpMoveRelType):
        self.EnableParameterWrite()
        if type == TcpMoveRelType.Rotation:
            self.__set_int_value_to_address(AddressTable.TCP_Relative_Type,0)
        elif type == TcpMoveRelType.Translation:
            self.__set_int_value_to_address(AddressTable.TCP_Relative_Type,1)
        else:
            print("TcpMove Type Error")
        self.DisableParameterWrite()

    def ExecuteMoveLinear(self):
        self.__set_bool_value_to_address(AddressTable.Move_Linear_Execute,False)
        sleep(0.1)
        self.__set_bool_value_to_address(AddressTable.Move_Linear_Execute,True)
        sleep(0.1)
        self.__set_bool_value_to_address(AddressTable.Move_Linear_Execute,False)
        sleep(0.1)


    def AbortMoveLinear(self):
        self.__set_bool_value_to_address(AddressTable.Move_Linear_Abort,True)
        sleep(0.1)
        self.__set_bool_value_to_address(AddressTable.Move_Linear_Abort,False)
        sleep(0.1)


    def ExecuteMoveJoint(self):
        self.__set_bool_value_to_address(AddressTable.Move_Joint_Execute,True)
        sleep(0.1)
        self.__set_bool_value_to_address(AddressTable.Move_Joint_Execute,False)

    def AbortMoveJoint(self):
        self.__set_bool_value_to_address(AddressTable.Move_Joint_Abort,True)
        sleep(0.1)
        self.__set_bool_value_to_address(AddressTable.Move_Joint_Abort,True)

    def ExecuteMovePathJoint(self):
        self.__set_bool_value_to_address(AddressTable.Move_Path_Joint_Execute,True)
        sleep(0.1)
        self.__set_bool_value_to_address(AddressTable.Move_Path_Joint_Execute, False)

    def AbortMovePathJoint(self):
        self.__set_bool_value_to_address(AddressTable.Move_Path_Joint_Abort,True)
        sleep(0.1)
        self.__set_bool_value_to_address(AddressTable.Move_Path_Joint_Abort, False)

    def ExecuteMovePathLinear(self):
        self.__set_bool_value_to_address(AddressTable.Move_Path_Linear_Execute,True)
        sleep(0.1)
        self.__set_bool_value_to_address(AddressTable.Move_Path_Linear_Execute, False)

    def AbortMovePathLinear(self):
        self.__set_bool_value_to_address(AddressTable.Move_Path_Linear_Abort,True)
        sleep(0.1)
        self.__set_bool_value_to_address(AddressTable.Move_Path_Linear_Abort, False)

    def ExecuteTCPMoveRel(self):
        self.__set_bool_value_to_address(AddressTable.TCP_Move_Rel_Execute,True)
        sleep(0.1)
        self.__set_bool_value_to_address(AddressTable.TCP_Move_Rel_Execute, False)

    def AbortTCPMoveRel(self):
        self.__set_bool_value_to_address(AddressTable.TCP_Move_Rel_Abort,True)
        sleep(0.1)
        self.__set_bool_value_to_address(AddressTable.TCP_Move_Rel_Abort, False)

    def ExecuteJointMoveABS(self,joint_index:int):
        if joint_index == 1:
            self.__set_bool_value_to_address(AddressTable.Joint1_MoveABS_Execute,True)
            sleep(0.1)
            self.__set_bool_value_to_address(AddressTable.Joint1_MoveABS_Execute, False)
        elif joint_index == 2:
            self.__set_bool_value_to_address(AddressTable.Joint2_MoveABS_Execute, True)
            sleep(0.1)
            self.__set_bool_value_to_address(AddressTable.Joint2_MoveABS_Execute, False)
        elif joint_index == 3:
            self.__set_bool_value_to_address(AddressTable.Joint3_MoveABS_Execute, True)
            sleep(0.1)
            self.__set_bool_value_to_address(AddressTable.Joint3_MoveABS_Execute, False)
        elif joint_index == 4:
            self.__set_bool_value_to_address(AddressTable.Joint4_MoveABS_Execute, True)
            sleep(0.1)
            self.__set_bool_value_to_address(AddressTable.Joint4_MoveABS_Execute, False)
        elif joint_index == 5:
            self.__set_bool_value_to_address(AddressTable.Joint5_MoveABS_Execute, True)
            sleep(0.1)
            self.__set_bool_value_to_address(AddressTable.Joint5_MoveABS_Execute, False)
        elif joint_index == 6:
            self.__set_bool_value_to_address(AddressTable.Joint6_MoveABS_Execute, True)
            sleep(0.1)
            self.__set_bool_value_to_address(AddressTable.Joint6_MoveABS_Execute, False)
        else:
            print("index Error")

    def StopJointMoveAbs(self,joint_index:int):
        if joint_index == 1:
            self.__set_bool_value_to_address(AddressTable.Joint1_Stop_Execute,True)
            sleep(0.1)
            self.__set_bool_value_to_address(AddressTable.Joint1_Stop_Execute, False)
        elif joint_index == 2:
            self.__set_bool_value_to_address(AddressTable.Joint2_Stop_Execute, True)
            sleep(0.1)
            self.__set_bool_value_to_address(AddressTable.Joint2_Stop_Execute, False)
        elif joint_index == 3:
            self.__set_bool_value_to_address(AddressTable.Joint3_Stop_Execute, True)
            sleep(0.1)
            self.__set_bool_value_to_address(AddressTable.Joint3_Stop_Execute, False)
        elif joint_index == 4:
            self.__set_bool_value_to_address(AddressTable.Joint4_Stop_Execute, True)
            sleep(0.1)
            self.__set_bool_value_to_address(AddressTable.Joint4_Stop_Execute, False)
        elif joint_index == 5:
            self.__set_bool_value_to_address(AddressTable.Joint5_Stop_Execute, True)
            sleep(0.1)
            self.__set_bool_value_to_address(AddressTable.Joint5_Stop_Execute, False)
        elif joint_index == 6:
            self.__set_bool_value_to_address(AddressTable.Joint6_Stop_Execute, True)
            sleep(0.1)
            self.__set_bool_value_to_address(AddressTable.Joint6_Stop_Execute, False)
        else:
            print("index Error")


    def JogForward(self,joint_index:int,value:bool):
        if joint_index == 1:
            self.__set_bool_value_to_address(AddressTable.Jog_Forward_1, value)
            self.__set_bool_value_to_address(AddressTable.Jog_Backward_1, False)
        elif joint_index == 2:
            self.__set_bool_value_to_address(AddressTable.Jog_Forward_2, value)
            self.__set_bool_value_to_address(AddressTable.Jog_Backward_2, False)
        elif joint_index == 3:
            self.__set_bool_value_to_address(AddressTable.Jog_Forward_3, value)
            self.__set_bool_value_to_address(AddressTable.Jog_Backward_3, False)
        elif joint_index == 4:
            self.__set_bool_value_to_address(AddressTable.Jog_Forward_4, value)
            self.__set_bool_value_to_address(AddressTable.Jog_Backward_4, False)
        elif joint_index == 5:
            self.__set_bool_value_to_address(AddressTable.Jog_Forward_5, value)
            self.__set_bool_value_to_address(AddressTable.Jog_Backward_5, False)
        elif joint_index == 6:
            self.__set_bool_value_to_address(AddressTable.Jog_Forward_6, value)
            self.__set_bool_value_to_address(AddressTable.Jog_Backward_6, False)
        else:
            print("index Error")

    def JogBackward(self,joint_index:int,value:bool):
        if joint_index == 1:
            self.__set_bool_value_to_address(AddressTable.Jog_Backward_1, value)
            self.__set_bool_value_to_address(AddressTable.Jog_Forward_1, False)
        elif joint_index == 2:
            self.__set_bool_value_to_address(AddressTable.Jog_Backward_2, value)
            self.__set_bool_value_to_address(AddressTable.Jog_Forward_2, False)
        elif joint_index == 3:
            self.__set_bool_value_to_address(AddressTable.Jog_Backward_3, value)
            self.__set_bool_value_to_address(AddressTable.Jog_Forward_3, False)
        elif joint_index == 4:
            self.__set_bool_value_to_address(AddressTable.Jog_Backward_4, value)
            self.__set_bool_value_to_address(AddressTable.Jog_Forward_4, False)
        elif joint_index == 5:
            self.__set_bool_value_to_address(AddressTable.Jog_Backward_5, value)
            self.__set_bool_value_to_address(AddressTable.Jog_Forward_5, False)
        elif joint_index == 6:
            self.__set_bool_value_to_address(AddressTable.Jog_Backward_6, value)
            self.__set_bool_value_to_address(AddressTable.Jog_Forward_6, False)
        else:
            print("index Error")

    def SetJogMod(self,Mode:JogMode):
        self.EnableParameterWrite()
        
        if Mode == JogMode.DistanceJog:
            self.__set_int_value_to_address(AddressTable.Jog_Mode,1)
        elif Mode == JogMode.VelocityJog:
            self.__set_int_value_to_address(AddressTable.Jog_Mode,0)
        else:
            print("JogMod Error")

        self.DisableParameterWrite()



    def RobotEnable(self):
        self.__set_bool_value_to_address(AddressTable.Power_Enable,True)

    def RobotDisable(self):
        self.__set_bool_value_to_address(AddressTable.Power_Enable,False)

    def RobotReset(self):
        self.__set_bool_value_to_address(AddressTable.Reset_Execute,True)
        sleep(0.1)
        self.__set_bool_value_to_address(AddressTable.Reset_Execute,False)

    def RobotStop(self):
        self.__set_bool_value_to_address(AddressTable.Stop_Execute,True)
        sleep(0.1)
        self.__set_bool_value_to_address(AddressTable.Stop_Execute,False)

    def AirLockControl(self,cmd:AirLock):
        if cmd == AirLock.Hold:
            self.__set_int_value_to_address(AddressTable.AirLock,0)
        elif cmd == AirLock.On:
            self.__set_int_value_to_address(AddressTable.AirLock,1)
        elif cmd == AirLock.Off:
            self.__set_int_value_to_address(AddressTable.AirLock,2)
        

    def RobotSetParameters(self,parameters:RobotParameters):
        self.EnableParameterWrite()
        self.__set_double_value_to_address(AddressTable.DH_D1,parameters.DH_D1)
        self.__set_double_value_to_address(AddressTable.DH_D3,parameters.DH_D3)
        self.__set_double_value_to_address(AddressTable.DH_D6,parameters.DH_D6)

        self.__set_double_value_to_address(AddressTable.MoveJ_Refference_Velocity, parameters.MoveJ_Refference_Velocity)
        self.__set_double_value_to_address(AddressTable.MoveJ_Refference_Acceleration, parameters.MoveJ_Refference_Acceleration)
        self.__set_double_value_to_address(AddressTable.MoveJ_Refference_Deceleration, parameters.MoveJ_Refference_Deceleration)

        self.__set_double_value_to_address(AddressTable.Jog_IncDistance_1,parameters.Jog_IncDistance_1)
        self.__set_double_value_to_address(AddressTable.Jog_IncDistance_2,parameters.Jog_IncDistance_2)
        self.__set_double_value_to_address(AddressTable.Jog_IncDistance_3,parameters.Jog_IncDistance_3)
        self.__set_double_value_to_address(AddressTable.Jog_IncDistance_4,parameters.Jog_IncDistance_4)
        self.__set_double_value_to_address(AddressTable.Jog_IncDistance_5,parameters.Jog_IncDistance_5)
        self.__set_double_value_to_address(AddressTable.Jog_IncDistance_6,parameters.Jog_IncDistance_6)

        self.__set_double_value_to_address(AddressTable.Jog_Velocity_1,parameters.Jog_Velocity_1)
        self.__set_double_value_to_address(AddressTable.Jog_Velocity_2,parameters.Jog_Velocity_2)
        self.__set_double_value_to_address(AddressTable.Jog_Velocity_3,parameters.Jog_Velocity_3)
        self.__set_double_value_to_address(AddressTable.Jog_Velocity_4,parameters.Jog_Velocity_4)
        self.__set_double_value_to_address(AddressTable.Jog_Velocity_5,parameters.Jog_Velocity_5)
        self.__set_double_value_to_address(AddressTable.Jog_Velocity_6,parameters.Jog_Velocity_6)
        # 下位机Jog模块中还有三个参数未分配地址

        self.__set_double_value_to_address(AddressTable.MoveL_Refference_Linear_Velocity,parameters.MoveL_Reference_Linear_Velocity)
        self.__set_double_value_to_address(AddressTable.MoveL_Refference_Linear_Acceleration,parameters.MoveL_Reference_Linear_Acceleration)
        self.__set_double_value_to_address(AddressTable.MoveL_Refference_Linear_Deceleration,parameters.MoveL_Reference_Linear_Deceleration)
        self.__set_double_value_to_address(AddressTable.MoveL_Refference_Angular_Velocity,parameters.MoveL_Reference_Angular_Velocity)
        self.__set_double_value_to_address(AddressTable.MoveL_Refference_Angular_Acceleration,parameters.MoveL_Reference_Angular_Acceleration)
        self.__set_double_value_to_address(AddressTable.MoveL_Refference_Angular_Deceleration,parameters.MoveL_Reference_Angular_Deceleration)

        self.__set_double_value_to_address(AddressTable.Joint1_Refference_Velocity,parameters.Joint1_Reference_Velocity)
        self.__set_double_value_to_address(AddressTable.Joint2_Refference_Velocity,parameters.Joint2_Reference_Velocity)
        self.__set_double_value_to_address(AddressTable.Joint3_Refference_Velocity,parameters.Joint3_Reference_Velocity)
        self.__set_double_value_to_address(AddressTable.Joint4_Refference_Velocity,parameters.Joint4_Reference_Velocity)
        self.__set_double_value_to_address(AddressTable.Joint5_Refference_Velocity,parameters.Joint5_Reference_Velocity)
        self.__set_double_value_to_address(AddressTable.Joint6_Refference_Velocity,parameters.Joint6_Reference_Velocity)

        self.__set_double_value_to_address(AddressTable.Joint1_Refference_Acceleration,parameters.Joint1_Reference_Acceleration)
        self.__set_double_value_to_address(AddressTable.Joint2_Refference_Acceleration,parameters.Joint2_Reference_Acceleration)
        self.__set_double_value_to_address(AddressTable.Joint3_Refference_Acceleration,parameters.Joint3_Reference_Acceleration)
        self.__set_double_value_to_address(AddressTable.Joint4_Refference_Acceleration,parameters.Joint4_Reference_Acceleration)
        self.__set_double_value_to_address(AddressTable.Joint5_Refference_Acceleration,parameters.Joint5_Reference_Acceleration)
        self.__set_double_value_to_address(AddressTable.Joint6_Refference_Acceleration,parameters.Joint6_Reference_Acceleration)

        self.__set_double_value_to_address(AddressTable.Joint1_Refference_Jerk,parameters.Joint1_Reference_Jerk)
        self.__set_double_value_to_address(AddressTable.Joint2_Refference_Jerk,parameters.Joint2_Reference_Jerk)
        self.__set_double_value_to_address(AddressTable.Joint3_Refference_Jerk,parameters.Joint3_Reference_Jerk)
        self.__set_double_value_to_address(AddressTable.Joint4_Refference_Jerk,parameters.Joint4_Reference_Jerk)
        self.__set_double_value_to_address(AddressTable.Joint5_Refference_Jerk,parameters.Joint5_Reference_Jerk)
        self.__set_double_value_to_address(AddressTable.Joint6_Refference_Jerk,parameters.Joint6_Reference_Jerk)

        self.DisableParameterWrite()

    def GetRobotState(self)->RobotStatus:
        try:
            discrete_Inputbytes, Input_RegisterBytes, Holding_RegisterBytes = self.__get_all_remote_data_registers()
            success = True
        except Exception as ex:
            success = False
            print(ex)
        if not success:
            print("GetRobotState failed")
            return None
        else:
            status = RobotStatus()
            status.Joints.J1.ActualPosition = self.utils.get_double_value_from_bytes(Input_RegisterBytes,AddressTable.Joint1_Actual_Position)
            status.Joints.J2.ActualPosition = self.utils.get_double_value_from_bytes(Input_RegisterBytes,AddressTable.Joint2_Actual_Position)
            status.Joints.J3.ActualPosition = self.utils.get_double_value_from_bytes(Input_RegisterBytes,AddressTable.Joint3_Actual_Position)
            status.Joints.J4.ActualPosition = self.utils.get_double_value_from_bytes(Input_RegisterBytes,AddressTable.Joint4_Actual_Position)
            status.Joints.J5.ActualPosition = self.utils.get_double_value_from_bytes(Input_RegisterBytes,AddressTable.Joint5_Actual_Position)
            status.Joints.J6.ActualPosition = self.utils.get_double_value_from_bytes(Input_RegisterBytes,AddressTable.Joint6_Actual_Position)

            status.Joints.J1.ActualVelocity = self.utils.get_double_value_from_bytes(Input_RegisterBytes,AddressTable.Joint1_Actual_Velocity)
            status.Joints.J2.ActualVelocity = self.utils.get_double_value_from_bytes(Input_RegisterBytes,AddressTable.Joint2_Actual_Velocity)
            status.Joints.J3.ActualVelocity = self.utils.get_double_value_from_bytes(Input_RegisterBytes,AddressTable.Joint3_Actual_Velocity)
            status.Joints.J4.ActualVelocity = self.utils.get_double_value_from_bytes(Input_RegisterBytes,AddressTable.Joint4_Actual_Velocity)
            status.Joints.J5.ActualVelocity = self.utils.get_double_value_from_bytes(Input_RegisterBytes,AddressTable.Joint5_Actual_Velocity)
            status.Joints.J6.ActualVelocity = self.utils.get_double_value_from_bytes(Input_RegisterBytes,AddressTable.Joint6_Actual_Velocity)

            status.Joints.J1.ActualAcceleration = self.utils.get_double_value_from_bytes(Input_RegisterBytes,AddressTable.Joint1_Actual_Acceleration)
            status.Joints.J2.ActualAcceleration = self.utils.get_double_value_from_bytes(Input_RegisterBytes,AddressTable.Joint2_Actual_Acceleration)
            status.Joints.J3.ActualAcceleration = self.utils.get_double_value_from_bytes(Input_RegisterBytes,AddressTable.Joint3_Actual_Acceleration)
            status.Joints.J4.ActualAcceleration = self.utils.get_double_value_from_bytes(Input_RegisterBytes,AddressTable.Joint4_Actual_Acceleration)
            status.Joints.J5.ActualAcceleration = self.utils.get_double_value_from_bytes(Input_RegisterBytes,AddressTable.Joint5_Actual_Acceleration)
            status.Joints.J6.ActualAcceleration = self.utils.get_double_value_from_bytes(Input_RegisterBytes,AddressTable.Joint6_Actual_Acceleration)

            status.TcpPose.X = self.utils.get_double_value_from_bytes(Input_RegisterBytes,AddressTable.TCP_Pose_X)
            status.TcpPose.Y = self.utils.get_double_value_from_bytes(Input_RegisterBytes,AddressTable.TCP_Pose_Y)
            status.TcpPose.Z = self.utils.get_double_value_from_bytes(Input_RegisterBytes,AddressTable.TCP_Pose_Z)
            status.TcpPose.Roll = self.utils.get_double_value_from_bytes(Input_RegisterBytes,AddressTable.TCP_Pose_Roll)
            status.TcpPose.Pitch = self.utils.get_double_value_from_bytes(Input_RegisterBytes,AddressTable.TCP_Pose_Pitch)
            status.TcpPose.Yaw = self.utils.get_double_value_from_bytes(Input_RegisterBytes,AddressTable.TCP_Pose_Yaw)

            status.Manage.Error = self.utils.get_bit_value_from_bytes(discrete_Inputbytes,AddressTable.Error)
            status.Manage.Mode = self.utils.get_short_value_from_bytes(Holding_RegisterBytes,AddressTable.Ctrl_Mode)
            status.Manage.Enabled = self.utils.get_bit_value_from_bytes(discrete_Inputbytes,AddressTable.Enabled)
            status.Manage.Moving = self.utils.get_bit_value_from_bytes(discrete_Inputbytes,AddressTable.Moving)
            status.AirLock = self.utils.get_short_value_from_bytes(Holding_RegisterBytes,AddressTable.AirLock)

            return status


    def SetDigitalOutput(self, index: int, value: bool):
        self.lock.acquire()
        try:
            self.client.write_coil(AddressTable.DigitalOutPut_0 + index, value)
        except Exception as ex:
            print(ex)
        finally:
            self.lock.release()


    # 读取力传感器的数据,返回一个list,FT
    def GetFTSensorData(self):
        try:
            self.lock.acquire()
            FT_Input_RegisterBytes_1 = self.client.read_input_registers(151,24).registers
            self.lock.release()
            success = True
        except Exception as ex:
            success = False
            print(ex)
        if not success:
            print("Get FT Sensor Data Failed")
            return None
        else:
            FT = [0]*6
            FT[0] = self.utils.get_double_value_from_bytes(FT_Input_RegisterBytes_1, AddressTable.Fx - 151)
            FT[1] = self.utils.get_double_value_from_bytes(FT_Input_RegisterBytes_1, AddressTable.Fy - 151)
            FT[2] = self.utils.get_double_value_from_bytes(FT_Input_RegisterBytes_1, AddressTable.Fz - 151)
            FT[3] = self.utils.get_double_value_from_bytes(FT_Input_RegisterBytes_1, AddressTable.Tx - 151)
            FT[4] = self.utils.get_double_value_from_bytes(FT_Input_RegisterBytes_1, AddressTable.Ty - 151)
            FT[5] = self.utils.get_double_value_from_bytes(FT_Input_RegisterBytes_1, AddressTable.Tz - 151)
            return FT


    # 力传感器数据软标定
    def FtSensorSoftCalibration(self):
        self.__set_bool_value_to_address(AddressTable.FTSensor_Software_Execute,True)
        sleep(0.1)
        self.__set_bool_value_to_address(AddressTable.FTSensor_Software_Execute,False)