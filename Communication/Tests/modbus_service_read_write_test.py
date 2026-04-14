# modbus_service_read_write_test.py
import os
import asyncio
from Communication.ModBusCommunicator import ModBusCommunicator
from Communication.CompactEntry import AddressBook
from Communication.ModBusService import ModBusService



JSON_PATH = os.path.join("Communication", "modbus_address_book.compact_win.json")

# 连接参数
# HOST = '192.168.232.155'
HOST = '10.69.160.155'
PORT = 502
UNIT = 1
POLL_INTERVAL = 0.04  # 服务轮询间隔（与ModBusService默认一致）


async def main():
    # 验证地址簿文件存在性
    print(f"地址簿 JSON 路径：{JSON_PATH}")
    print(f"地址簿文件是否存在：{os.path.exists(JSON_PATH)}")
    if not os.path.exists(JSON_PATH):
        print("地址簿文件不存在，退出测试")
        return

    # 1. 加载地址簿（转换为CompactEntry字典）
    address_book = AddressBook.load(JSON_PATH)
    # 打印整个地址簿以供调试
    # print("加载的地址簿内容：")
    # for name, entry in address_book.items():
    #     print(f"  {name}: {entry}")

    if not address_book:
        print("地址簿加载失败，退出测试")
        return

    # 2. 创建ModBusCommunicator（服务依赖的通信器）
    communicator = ModBusCommunicator(
        host=HOST,
        port=PORT,
        unit_id=UNIT,
        swap_words=True  # 保持与原脚本一致的字节序设置
    )

    # 3. 初始化ModBusService（传入通信器和地址簿）
    service = ModBusService(
        communicator=communicator,
        address_book=address_book,
        poll_interval=POLL_INTERVAL
    )

    # 4. 启动服务（内部处理连接和队列初始化）
    print(f"启动服务，连接到 {HOST}:{PORT} ...")
    try:
        await service.start()
        if not service.is_running:
            print("服务启动失败，退出测试")
            return
        print("服务启动成功，等待初始化完成...")
        await asyncio.sleep(1.0)  # 给服务预留启动和首次轮询的时间
    except Exception as e:
        print(f"服务启动出错：{e}")
        return

    try:
        # 5. 执行读写测试（与原同步脚本测试内容一一对应）

        # 5.1 读取线圈（Instructions类变量）
        try:
            power_on = await service.read_bool('Instructions.Power_On')
        except Exception as e:
            power_on = f"读取错误: {str(e)}"
        print(f"Instructions.Power_On = {power_on}")

        try:
            reset_ft = await service.read_bool('Instructions.Reset_FTSensor_Calibration')
        except Exception as e:
            reset_ft = f"读取错误: {str(e)}"
        print(f"Instructions.Reset_FTSensor_Calibration = {reset_ft}")


        # 5.2 读取Flags类布尔变量
        try:
            initialized = await service.read_bool('Flags.Initialized')
        except Exception as e:
            initialized = f"读取错误: {str(e)}"
        print(f"Flags.Initialized = {initialized}")


        # 5.3 读取Flags.Joint_Moving数组（1-based索引）
        jm_vals = []
        try:
            # 从地址簿获取数组维度（假设dims[0]为数组长度）
            jm_entry = AddressBook._get_entry_case_insensitive(address_book, 'Flags.Joint_Moving')
            if jm_entry and jm_entry.dims:
                for i in range(1, jm_entry.dims[0] + 1):  # 索引从1开始（适配Codesys）
                    val = await service.read_bool('Flags.Joint_Moving', i)
                    jm_vals.append(val)
            else:
                jm_vals = ["未找到数组维度信息"]
        except Exception as e:
            jm_vals = [f"数组读取错误: {str(e)}"]
        print(f"Flags.Joint_Moving = {jm_vals}")


        # 5.4 读取Parameters.DH_Parameters二维数组 [6,4]
        try:
            dh_entry = AddressBook._get_entry_case_insensitive(address_book, 'Parameters.DH_Parameters')
            if dh_entry and dh_entry.dims and len(dh_entry.dims) == 2:
                print("Parameters.DH_Parameters 二维数组:")
                for i in range(1, dh_entry.dims[0] + 1):  # 行索引（1-based）
                    for j in range(1, dh_entry.dims[1] + 1):  # 列索引（1-based）
                        val = await service.read_real('Parameters.DH_Parameters', i, j)
                        print(f"  [{i},{j}] = {val:.4f}")
            else:
                print("Parameters.DH_Parameters: 未找到二维数组维度信息")
        except Exception as e:
            print(f"Parameters.DH_Parameters 读取错误: {str(e)}")

        # 写入并验证Parameters.DH_Parameters
        DHParameters = [[ 0.0, 0.0, 157.8301, 0.0 ],
            [ 0.0, 1.570796, 0.0, 0 ],
            [ -270.1666, 0.0, 0.0, 0 ],
            [ -270.506, 0, 131.1573, 0.0 ],
            [ 0.0, 1.570796, 108.4209, 0.0 ],
            [ 0.0, -1.570796, 68.5983, 0.0 ]]
        try:
            print("\n写入 Parameters.DH_Parameters 二维数组:")
            for i in range(1, dh_entry.dims[0] + 1):  # 行索引（1-based）
                for j in range(1, dh_entry.dims[1] + 1):  # 列索引（1-based）
                    val_to_write = DHParameters[i-1][j-1]
                    await service.write_real('Parameters.DH_Parameters', val_to_write, i, j)
                    print(f"  写入 [{i},{j}] = {val_to_write:.4f}")
            await asyncio.sleep(0.5)  # 等待写入生效
            # 读取验证
            print("读取验证 Parameters.DH_Parameters 二维数组:")
            for i in range(1, dh_entry.dims[0] + 1):  # 行索引（1-based）
                for j in range(1, dh_entry.dims[1] + 1):  # 列索引（1-based）
                    val_read = await service.read_real('Parameters.DH_Parameters', i, j)
                    print(f"  读取 [{i},{j}] = {val_read:.4f}")
        except Exception as e:
            print(f"Parameters.DH_Parameters 写入/验证错误: {str(e)}")
            


        # 5.5 读取Parameters.Analog_Output数组 [8]
        try:
            ao_entry = AddressBook._get_entry_case_insensitive(address_book, 'Parameters.Analog_Ouput')
            if ao_entry and ao_entry.dims:
                print("Parameters.Analog_Ouput 数组:")
                for i in range(1, ao_entry.dims[0] + 1):  # 1-based索引
                    val = await service.read_real('Parameters.Analog_Ouput', i)
                    print(f"  [{i}] = {val:.4f}")
            else:
                print("Parameters.Analog_Ouput: 未找到数组维度信息")
        except Exception as e:
            print(f"Parameters.Analog_Ouput 读取错误: {str(e)}")

        # 5.5.2 读Parameters.TCP_Refference_Angular_Deceleration（REAL类型）
        try:
            tcp_ref_dec = await service.read_real('Parameters.TCP_Refference_Angular_Deceleration')
            print(f"Parameters.TCP_Refference_Angular_Deceleration = {tcp_ref_dec:.4f}")
        except Exception as e:
            tcp_ref_dec = f"读取错误: {str(e)}"


        # 5.6 读取Status.Error_ID（DWORD类型）
        try:
            error_id = await service.read_dword('Status.Error_ID')
        except Exception as e:
            error_id = f"读取错误: {str(e)}"
        print(f"Status.Error_ID = {error_id}")


        # 5.7 读取Status.TCP_Wrench数组 [6]
        try:
            tw_entry = AddressBook._get_entry_case_insensitive(address_book, 'Status.TCP_Wrench')
            if tw_entry and tw_entry.dims:
                print("Status.TCP_Wrench 数组:")
                for i in range(1, tw_entry.dims[0] + 1):  # 1-based索引
                    val = await service.read_real('Status.TCP_Wrench', i)
                    print(f"  [{i}] = {val:.4f}")
            else:
                print("Status.TCP_Wrench: 未找到数组维度信息")
        except Exception as e:
            print(f"Status.TCP_Wrench 读取错误: {str(e)}")


        # # 5.8 写入并验证Parameters.Movement_Mode（UINT类型）
        # try:
        #     target_mode = 2  # 目标模式值（与原同步脚本一致）
        #     print(f"\n写入 Parameters.Movement_Mode = {target_mode}")
        #     await service.write_uint('Parameters.Movement_Mode', target_mode)
        #     await asyncio.sleep(0.5)  # 等待写入生效（服务轮询需要时间）

        #     # 读取验证
        #     read_back = await service.read_uint('Parameters.Movement_Mode')
        #     print(f"写入后读取 Parameters.Movement_Mode = {read_back}")
        #     if read_back == target_mode:
        #         print("✅ 写入验证成功")
        #     else:
        #         print("❌ 写入验证失败（值不匹配）")
        # except Exception as e:
        #     print(f"Parameters.Movement_Mode 操作错误: {str(e)}")


    finally:
        # 6. 停止服务（内部处理断开连接和资源清理）
        print("\n停止服务...")
        try:
            await service.stop()
            print("服务已停止，测试结束")
        except Exception as e:
            print(f"服务停止出错: {e}")


if __name__ == '__main__':
    # 运行异步测试
    asyncio.run(main())