# -*- coding: utf-8 -*-
# 脑电原始数据解析器,解析单行数据
import csv

# 原始数据样例，格式为：2字节的同步标识，1字节的载荷长度，32字节的有效载荷，1字节的校验和，共36字节
RAW_DATA_EXAMPLE = [
    'aaaa2002c883180132140026f70014c7000bfa0007a9001ee200064b00007704000500d5',
    'aaaa2002008318008e9b0114100070cb0041ed002192001347000898000617042c053676',
    'aaaa200200831801836a00701400046300082e000ab4000c0600047500023a043005365f',
    'aaaa2002008318007363006189000a5e000fec001b8c0019710011af000845043f052b8e',
    'aaaa2002008318002c95001949002609000b6f001c2a000bd20009ce0003430442052cdf',
    'aaaa2002008318000b670013f90018bd000689000c540012d9000ad20001dd0458052feb',
]


def parse_packet(raw_data):
    """
    解析一行原始数据
    :param row_data: 72  [0:4]同步标识符  [4:6]载荷长度  [6:-2]载荷   [-2:]校验和
    :return:
    """
    sync = raw_data[0:4]
    plength = raw_data[4:6]
    payload = raw_data[6:-2]
    checksum = raw_data[-2:]

    # 验证载荷长度
    assert int(plength, 16) * 2 < 169, '载荷长度超出阈值'
    assert int(plength, 16) * 2 == len(payload), '载荷长度不一致'

    # print(int('aa', 16))

    # 验证校验和
    if (int(checksum, 16) == calculate_checksum(payload)):
        result = parse_payload(payload)
        return result
        # assert int(checksum, 16) == calculate_checksum(payload), '校验和不一致'
    else:
        # print('校验和错误')
        return None
    # # 解析字段
    # result = parse_payload(payload)
    # return result


def parse_payload(payload):
    """
    解析有效载荷
    :param payload:
    :return:
    """
    results = {}
    while len(payload) > 0:
        code = int(payload[0:2], 16)
        payload = payload[2:]
        if 0x02 <= code <= 0x07:
            result, payload = parse_single_byte(code, payload)
            results.update(result)
        else:
            result, payload = parse_multi_byte(code, payload)
            results.update(result)
    # print(results)

    return results


def parse_single_byte(code, payload):
    """
    解析单字节数据
    :param code: 字节码 0x02-0x07
    :param payload:
    :return:
    """
    key = ""
    if code == 0x02:
        # 解析信号质量， 0-255，越小质量越好
        key = "signal"

    elif code == 0x04:
        # attention, 0-100
        key = 'attention'

    elif code == 0x05:
        # meditation, 0-100
        key = 'meditation'

    value = int(payload[0:2], 16)
    result = {key: value}
    payload = payload[2:]
    return result, payload


def parse_multi_byte(code, payload):
    """
    解析多字节数据
    :param code: 0x80，0x81，0x83，0x86，0x55
    :param payload:
    :return:
    """
    key = ""
    if code == 0x83:
        length = 2 * int(payload[0:2], 16)
        payload = payload[2:]
        result = parse_eeg_band_ascii(payload[:length])
        payload = payload[length:]
    else:
        result = {
            'delta': 0,
            'theta': 0,
            'low_alpha': 0,
            'high_alpha': 0,
            'low_beta': 0,
            'high_beta': 0,
            'low_gama': 0,
            'mid_gama': 0,
        }
        length = 2 * int(payload[0:2], 16)
        payload = payload[2:]
        payload = payload[length:]

    return result, payload


def parse_eeg_band_ascii(eeg):
    """
    解析48位ascii码，共8个波段，每个波段3字节
    :param eeg:
    :return:
    """

    assert len(eeg) == 48, 'egg数据长度错误'

    # result = [{'delta': int(eeg[0:6], 16)},
    #           {'theta': int(eeg[6:12], 16)},
    #           {'low_alpha': int(eeg[12:18], 16)},
    #           {'high_alpha': int(eeg[18:24], 16)},
    #           {'low_beta': int(eeg[24:30], 16)},
    #           {'high_beta': int(eeg[30:36], 16)},
    #           {'low_gama': int(eeg[36:42], 16)},
    #           {'mid_gama': int(eeg[42:48], 16)}]

    result = {
        'delta': int(eeg[0:6], 16),
        'theta': int(eeg[6:12], 16),
        'low_alpha': int(eeg[12:18], 16),
        'high_alpha': int(eeg[18:24], 16),
        'low_beta': int(eeg[24:30], 16),
        'high_beta': int(eeg[30:36], 16),
        'low_gama': int(eeg[36:42], 16),
        'mid_gama': int(eeg[42:48], 16),
    }

    return result


def calculate_checksum(payload):
    """
    计算有效载荷的校验和
    :param payload:
    :return:
    """
    # 1.计算所有字节的和

    bytes_data = bytes.fromhex(payload)

    bytes_sum = 0
    for byte in bytes_data:
        bytes_sum += byte

    # 2.取低8位的值
    low_8_bits = bytes_sum & 0xFF

    # 3.低8位按位取反
    checksum = ~low_8_bits & 0xFF

    return checksum


def classify_attention_level(value):
    """根据数值返回注意力等级"""
    value = int(value)

    if value == 0:
        return "无效"
    elif 1 <= value <= 20:
        return "低"
    elif 20 < value <= 40:
        return "稍低"
    elif 40 < value <= 60:
        return "中性"
    elif 60 < value <= 80:
        return "稍高"
    elif 80 < value <= 100:
        return "高"
    else:
        return "异常值"  # 处理超出范围的情况


def process_eeg_file(input_file_path, output_file_path):
    """
    处理EEG数据文件
    :param input_file_path: 输入文件路径
    :param output_file_path: 输出文件路径
    """
    results = []

    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # 跳过空行
                    continue

                try:
                    # 分割时间戳和数据
                    if ',' in line:
                        timestamp_str, raw_data = line.split(',', 1)
                        raw_data = raw_data.strip()

                        # 解析数据包
                        parsed_data = parse_packet(raw_data)

                        if parsed_data is not None:
                            # 获取attention值，如果不存在则设为None
                            attention_value = parsed_data.get('attention')
                            # 分类
                            level = classify_attention_level(attention_value)
                            # 将结果添加到列表
                            results.append({
                                'timestamp': timestamp_str,
                                'attention': level,
                                'raw_data': raw_data,
                                'parsed_data': parsed_data
                            })
                            # print(f"第 {line_num} 行处理成功: attention={attention_value}")
                        else:
                            print(f"第 {line_num} 行: 数据包解析失败 (校验和错误或数据无效)")
                            results.append({
                                'timestamp': timestamp_str,
                                'attention': '无效',
                                'raw_data': raw_data,
                                'parsed_data': None,
                                'error': '解析失败'
                            })
                    else:
                        print(f"第 {line_num} 行: 格式错误，缺少逗号分隔符")

                except Exception as e:
                    print(f"第 {line_num} 行处理出错: {type(e).__name__}: {e}")
                    if ',' in line:
                        timestamp_str, raw_data = line.split(',', 1)
                        results.append({
                            'timestamp': timestamp_str,
                            'attention': '无效',
                            'raw_data': raw_data,
                            'parsed_data': None,
                            'error': f"{type(e).__name__}: {str(e)[:50]}"
                        })

    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file_path}")
        return
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return

    # 写入输出文件 (CSV格式)
    try:
        with open(output_file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # 写入表头
            writer.writerow(['timestamp', 'attention'])

            # 写入数据
            for result in results:
                if result['attention'] == '无效':
                    continue
                writer.writerow([result['timestamp'], result['attention']])

        print(f"\n处理完成! 共处理 {len(results)} 行数据")
        print(f"结果已保存到: {output_file_path}")
    except Exception as e:
        print(f"写入输出文件时出错: {e}")
    # 统计各等级数量
    from collections import Counter
    levels = [line.get('attention') for line in results]
    counter = Counter(levels)

    print("\n等级分布统计:")
    for level, count in counter.items():
        print(f"  {level}: {count} 行 ({count / len(results) * 100:.1f}%)")

if __name__ == '__main__':
    # for data in RAW_DATA_EXAMPLE:
    #     parse_packet(data)
    # parse_packet(RAW_DATA_EXAMPLE[0])
    # 传入未计算的脑电文件，输出注意力高中低的文件（去除无效数据）
    process_eeg_file(r"E:\数据\20231229 计算机网络考试数据汇总\第6组\脑电\2021214398_张颖.txt", r"D:\GraduationProject\demo1\output\2021214398_张颖.txt")
