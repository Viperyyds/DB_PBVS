# codesys_read_write_vars.py
import os
import time
from Communication.ModBusCommunicator import ModBusCommunicator
from Communication.CompactEntry import AddressBook

# 地址簿路径也可简化（直接从项目根目录出发）
JSON_PATH = os.path.join("Communication", "modbus_address_book.compact_win.json")

# 打印路径，确认是否正确指向 SDKPython 和地址簿文件

print("地址簿 JSON 路径：", JSON_PATH)
print("地址簿文件是否存在：", os.path.exists(JSON_PATH))

# HOST = '192.168.1.105'
HOST = '192.168.232.165'

PORT = 502
UNIT = 1


def main():
    # load address book
    book = AddressBook.load(JSON_PATH)

    # swap_words=True to match Codesys register word order (diagnosed earlier)
    comm = ModBusCommunicator(HOST, PORT, unit_id=UNIT, swap_words=True)
    print(f'Connecting to {HOST}:{PORT} ...')
    ok = comm.connect()
    if not ok:
        print('Connect failed')
        return
    print('Connected')

    try:
        # Helper to read based on CompactEntry.baseType
        def read_value(key, *indices):
            entry = AddressBook._get_entry_case_insensitive(book, key)
            base = (entry.baseType or '').upper()
            # choose proper communicator method
            if base == 'BOOL':
                return comm.read_bool(book, key, *indices)
            if base in ('UINT', 'WORD'):
                return comm.read_uint(book, key, *indices)
            if base in ('REAL', 'FLOAT'):
                return comm.read_real(book, key, *indices)
            if base in ('DWORD', 'DINT', 'UDINT'):
                return comm.read_dword(book, key, *indices)
            # fallback by bytesPerElem
            if entry.bytesPerElem == 1:
                return comm.read_bool(book, key, *indices)
            if entry.bytesPerElem == 2:
                return comm.read_uint(book, key, *indices)
            if entry.bytesPerElem == 4:
                return comm.read_dword(book, key, *indices)
            raise ValueError(f'Unsupported type for {key}: {entry.baseType}')


        # 1. read a couple of coils
        try:
            v = read_value('Instructions.Power_On')
        except Exception as e:
            v = f'error: {e}'
        print('Read back Instructions.Power_On =', v)

        try:
            v = read_value('Instructions.Reset_FTSensor_Calibration')
        except Exception as e:
            v = f'error: {e}'
        print('Read back Instructions.Reset_FTSensor_Calibration =', v)

        # Read single boolean
        try:
            b = read_value('Flags.Initialized')
        except Exception as e:
            b = f'error: {e}'
        print('Flags.Initialized =', b)

        # Read array Flags.Joint_Moving (use dims to determine length, 1-based indices)
        jm_entry = AddressBook._get_entry_case_insensitive(book, 'Flags.Joint_Moving')
        jm_vals = []
        if jm_entry.dims:
            for i in range(1, jm_entry.dims[0] + 1):
                try:
                    v = read_value('Flags.Joint_Moving', i)
                except Exception as e:
                    v = f'error: {e}'
                jm_vals.append(v)
        print('Flags.Joint_Moving =', jm_vals)

        # Demonstrate reading arrays with proper indexing
        # Parameters.DH_Parameters is [6,4]
        dh_entry = AddressBook._get_entry_case_insensitive(book, 'Parameters.DH_Parameters')
        if dh_entry.dims and len(dh_entry.dims) == 2:
            for i in range(1, dh_entry.dims[0] + 1):
                for j in range(1, dh_entry.dims[1] + 1):
                    try:
                        f = read_value('Parameters.DH_Parameters', i, j)
                    except Exception as e:
                        f = f'error: {e}'
                    print(f'Read back Parameters.DH_Parameters[{i},{j}] =', f)
        # 写入DH参数
        # dh_writes = [(1, 3, 287), (3, 3,800), (6, 3, 245)]
        dh_writes = [(1, 3, 0), (3, 3, 0), (6, 3, 0)]
        # 实际写入的是DH_Parameters[1,3], [2,3],[3,3]
        for i, j, val in dh_writes:
            try:
                print(f'Writing Parameters.DH_Parameters[{i},{j}] = {val}')
                comm.write_real(book, 'Parameters.DH_Parameters', val, i, j)

                # comm.write_real(book, 'Parameters.DH_Parameters', val, i, j)
                time.sleep(0.1)  # small delay to ensure write is processed
                f = read_value('Parameters.DH_Parameters', i, j)
                print(f'Read back Parameters.DH_Parameters[{i},{j}] =', f)
            except Exception as e:
                print(f'Error writing Parameters.DH_Parameters[{i},{j}] =', e)

        # Parameters.LoadCOG dims[3]
        lc_entry = AddressBook._get_entry_case_insensitive(book, 'Parameters.LoadCOG')
        if lc_entry.dims:
            for i in range(1, lc_entry.dims[0] + 1):
                try:
                    f = read_value('Parameters.LoadCOG', i)
                except Exception as e:
                    f = f'error: {e}'
                print(f'Read back Parameters.LoadCOG[{i}] =', f)

        # Status.Error_ID (DWORD/uint)
        try:
            ui = read_value('Status.Error_ID')
        except Exception as e:
            ui = f'error: {e}'
        print('Read back Status.Error_ID =', ui)

        # Status.TCP_Wrench dims[6]
        tw_entry = AddressBook._get_entry_case_insensitive(book, 'Status.TCP_Wrench')
        if tw_entry.dims:
            for i in range(1, tw_entry.dims[0] + 1):
                try:
                    f = read_value('Status.TCP_Wrench', i)
                except Exception as e:
                    f = f'error: {e}'
                print(f'Read back Status.TCP_Wrench[{i}] =', f)

        # write Parameters.Movement_Mode (UINT/word)
        try:
            print('Writing Parameters.Movement_Mode = 2')
            comm.write_uint(book, 'Parameters.Movement_Mode', 2)
            time.sleep(0.1)  # small delay to ensure write is processed
            v = read_value('Parameters.Movement_Mode')
            print('Read back Parameters.Movement_Mode =', v)
        except Exception as e:
            print('Error writing Parameters.Movement_Mode =', e)

    finally:
        comm.disconnect()


if __name__ == '__main__':
    main()