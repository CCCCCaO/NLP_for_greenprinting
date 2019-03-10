import pandas as pd


def csv_preproc(ori_csv_dir, proc_csv_dir):
    """
    csv_preproc函数 用于对csv文件进行预处理操作：去除重复行，并按年份时间升序排列
    并将处理后的csv文件输出，可以用于检查

    参数：
        ori_csv_dir     原始csv文件的路径
        proc_csv_dir    输出处理后的csv文件的路径
    """
    try:
        csv_data = pd.read_csv(ori_csv_dir)
    except IOError:
        print("csv数据加载失败！请检查目录！")
    else:
        print("原始数据尺寸为：{}".format(csv_data.shape))
        # 在原地去重
        csv_data.drop_duplicates(inplace=True)
        print("去重后数据尺寸为：{}".format(csv_data.shape))
        # 按照年份排序
        csv_data_distinct_ordered = csv_data.sort_values(by='Date')
        # 输出新的csv文件
        csv_data_distinct_ordered.to_csv(proc_csv_dir, index=False)
        print('去除重复且按时间升序排列的csv文件输出完成！')
        print("---")


def csv_slice(proc_csv_dir, tar_csv_dir, start_year=1500, end_year=2019):
    """
    csv_slice函数 用于从去重且排序好的csv文件中 切出从start年份至end年的数据 用于分析
    
    参数：
        proc_csv_dir    去重且排序好的csv文件路径
        tar_csv_dir     选取某年到某年的csv文件并输出的路径
        start_year      起始年份 默认1500
        end_year        末尾年份 默认为2019
    """
    try:
        csv_data = pd.read_csv(proc_csv_dir)
    except IOError:
        print("处理后的csv数据加载失败！请检查目录！")
    else:
        print("读取数据尺寸为：{}".format(csv_data.shape))
        csv_xtoy = csv_data.loc[(csv_data['Date'] >= start_year) & (csv_data['Date'] <= end_year)]
        print("{}年到{}年 切片后的数据尺寸为：{}".format(start_year, end_year, csv_xtoy.shape))
        # 将结果输出至csv文件 不要列索引输出header取None 不要行索引index取False
        csv_xtoy[['Keywords', 'Abstract']].to_csv(tar_csv_dir, header=None, index=False)
        print("---")







ori_csv_dir = 'C:\\Users\\82460\\Documents\\GitHub\\green_printing\\ex_text.csv'
proc_csv_dir = 'C:\\Users\\82460\\Documents\\GitHub\\green_printing\\sliced_text\\proc_text.csv'
csv_preproc(ori_csv_dir, proc_csv_dir)
tar_csv_dir = 'C:\\Users\\82460\\Documents\\GitHub\\green_printing\\sliced_text\\text2000-2009.csv'
csv_slice(proc_csv_dir, tar_csv_dir,2000, 2009)