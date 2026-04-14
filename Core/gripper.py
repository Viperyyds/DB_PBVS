import serial
import time

class Gripper:
    """
    JODELL ERG32-150旋转电伺服电动夹爪控制类
    实现文档中所有功能（夹持/旋转控制、状态读取、故障处理等）
    """

    def __init__(self, port_name):
        self._serial_port = serial.Serial(
            port=port_name,
            baudrate=115200,
            bytesize=8,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1
        )
        self._slave_id = 0x09

    def open(self):
        if not self._serial_port.is_open:
            self._serial_port.open()
            return True
        return False

    def close(self):
        if self._serial_port.is_open:
            self._serial_port.close()
        return True

    def send_write_command(self, command: bytes):
        if not self._serial_port.is_open:
            raise Exception("串口未打开")
        self._serial_port.reset_input_buffer()
        self._serial_port.write(command)
        time.sleep(0.05)

    def send_read_command(self, command: bytes) -> bytes:
        if not self._serial_port.is_open:
            raise Exception("串口未打开")
        self._serial_port.reset_input_buffer()
        self._serial_port.write(command)
        time.sleep(0.05)
        response = self._serial_port.read(128)
        if len(response) == 0:
            raise TimeoutError("未收到响应")
        return response

    def calculate_crc(self, data: bytes) -> int:
        crc = 0xFFFF
        for b in data:
            crc ^= b
            for _ in range(8):
                if crc & 0x0001:
                    crc >>= 1
                    crc ^= 0xA001
                else:
                    crc >>= 1
        return crc

    # 夹持端控制
    def grip_reset(self):
        command = bytes([0x09, 0x10, 0x03, 0xE8, 0x00, 0x01, 0x02, 0x00, 0x00, 0xE5, 0xB8])
        self.send_write_command(command)

    def grip_activate(self):
        self.grip_reset()
        time.sleep(0.1)
        activate_cmd = bytes([0x09, 0x10, 0x03, 0xE8, 0x00, 0x01, 0x02, 0x00, 0x01, 0x24, 0x78])
        self.send_write_command(activate_cmd)
        count = 0
        while True:
            count += 1
            status = self.get_grip_status()
            print(f"Status: {status:02X}")
            if status == 0x18:
                break
            if count > 1000:
                return False
            time.sleep(0.05)
        return True

    def grip_ctrl(self, position, speed, force):
        command = bytearray(13)
        command[0] = 0x09
        command[1] = 0x10
        command[2] = 0x03
        command[3] = 0xEA
        command[4] = 0x00
        command[5] = 0x02
        command[6] = 0x04
        command[7] = speed
        command[8] = position
        command[9] = force
        command[10] = 0x01
        crc = self.calculate_crc(command[:11])
        command[11] = crc & 0xFF
        command[12] = (crc >> 8) & 0xFF
        self.send_write_command(command)
        while self.get_grip_position() < position:
            time.sleep(0.05)

    def grip_open_full(self, speed, force):
        self.grip_ctrl(0x00, speed, force)

    def grip_close_full(self, speed, force):
        self.grip_ctrl(0xFF, speed, force)

    def grip_close_full_no_param(self):
        command = bytes([0x09, 0x10, 0x03, 0xE8, 0x00, 0x01, 0x02, 0x04, 0x0B, 0xA6, 0xBF])
        self.send_write_command(command)
        time.sleep(1)

    def grip_open_full_no_param(self):
        command = bytes([0x09, 0x10, 0x03, 0xE8, 0x00, 0x01, 0x02, 0x03, 0x0B, 0xA4, 0x8F])
        self.send_write_command(command)
        time.sleep(1)

    def get_grip_status(self):
        command = bytes([0x09, 0x03, 0x07, 0xD0, 0x00, 0x01, 0x85, 0xCF])
        response = self.send_read_command(command)
        return (response[3] << 8) | response[4]

    def get_grip_position(self):
        command = bytes([0x09, 0x03, 0x07, 0xD2, 0x00, 0x02, 0x64, 0x0E])
        response = self.send_read_command(command)
        return response[4]

    def get_grip_fault(self):
        command = bytes([0x09, 0x03, 0x07, 0xD0, 0x00, 0x01, 0x85, 0xCF])
        response = self.send_read_command(command)
        return response[3]

    # 旋转端控制
    def rotate_reset(self):
        command = bytes([0x09, 0x10, 0x03, 0xE9, 0x00, 0x01, 0x02, 0x00, 0x00, 0xE4, 0x69])
        self.send_write_command(command)

    def rotate_activate(self):
        self.rotate_reset()
        time.sleep(0.1)
        activate_cmd = bytes([0x09, 0x10, 0x03, 0xE9, 0x00, 0x01, 0x02, 0x00, 0x01, 0x25, 0xA9])
        self.send_write_command(activate_cmd)
        count = 0
        while True:
            count += 1
            status = self.get_rotate_status()
            if status == 0x18:
                break
            if count > 1000:
                return False
            time.sleep(0.05)
        return True

    def rotate_absolute(self, angle, speed, torque):
        command = bytearray(17)
        command[0] = 0x09
        command[1] = 0x10
        command[2] = 0x03
        command[3] = 0xEC
        command[4] = 0x00
        command[5] = 0x04
        command[6] = 0x08
        command[7] = (angle >> 8) & 0xFF
        command[8] = angle & 0xFF
        command[9] = torque
        command[10] = speed
        command[11] = (angle >> 8) & 0xFF
        command[12] = angle & 0xFF
        command[13] = 0x00
        command[14] = 0x01
        crc = self.calculate_crc(command[:15])
        command[15] = crc & 0xFF
        command[16] = (crc >> 8) & 0xFF
        self.send_write_command(command)
        while abs(self.get_rotate_angle() - angle) > 5:
            time.sleep(0.05)

    def rotate_relative(self, angle, speed, torque):
        command = bytearray(17)
        command[0] = 0x09
        command[1] = 0x10
        command[2] = 0x03
        command[3] = 0xEC
        command[4] = 0x00
        command[5] = 0x04
        command[6] = 0x08
        command[7] = (angle >> 8) & 0xFF
        command[8] = angle & 0xFF
        command[9] = torque
        command[10] = speed
        command[11] = (angle >> 8) & 0xFF
        command[12] = angle & 0xFF
        command[13] = 0x00
        command[14] = 0x02
        crc = self.calculate_crc(command[:15])
        command[15] = crc & 0xFF
        command[16] = (crc >> 8) & 0xFF
        self.send_write_command(command)

    def rotate_clockwise_full_no_param(self):
        command = bytes([0x09, 0x10, 0x03, 0xE9, 0x00, 0x01, 0x02, 0x03, 0x0B, 0xA5, 0x5E])
        self.send_write_command(command)
        time.sleep(2)

    def rotate_counter_clockwise_full_no_param(self):
        command = bytes([0x09, 0x10, 0x03, 0xE9, 0x00, 0x01, 0x02, 0x04, 0x0B, 0xA7, 0x6E])
        self.send_write_command(command)
        time.sleep(2)

    def get_rotate_status(self):
        command = bytes([0x09, 0x03, 0x07, 0xD1, 0x00, 0x01, 0xD4, 0x0F])
        response = self.send_read_command(command)
        return (response[3] << 8) | response[4]

    def get_rotate_angle(self):
        command = bytes([0x09, 0x03, 0x07, 0xD4, 0x00, 0x04, 0x04, 0x0D])
        response = self.send_read_command(command)
        return (response[3] << 8) | response[4]

    def get_rotate_fault(self):
        command = bytes([0x09, 0x03, 0x07, 0xD1, 0x00, 0x01, 0xD4, 0x0F])
        response = self.send_read_command(command)
        return response[3]

    # 通用功能
    def get_device_id(self):
        command = bytes([0x09, 0x03, 0x13, 0x8A, 0x00, 0x01, 0x69, 0x6C])
        response = self.send_read_command(command)
        return response[4]

    def get_software_version(self):
        command = bytes([0x09, 0x03, 0x13, 0x89, 0x00, 0x01, 0x29, 0x68])
        response = self.send_read_command(command)
        return f"{response[3]}.{response[4]}"

    def get_bus_voltage(self):
        command = bytes([0x09, 0x03, 0x07, 0xD8, 0x00, 0x01, 0x55, 0xCB])
        response = self.send_read_command(command)
        return response[4]

    def get_temperature(self):
        command = bytes([0x09, 0x03, 0x07, 0xD8, 0x00, 0x01, 0x55, 0xCB])
        response = self.send_read_command(command)
        return response[3]

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    
