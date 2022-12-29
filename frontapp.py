import argparse
import time
import subprocess
import streamlit as st
from generate import gen_sample

def subprocess_getoutput(stmt):
    result = subprocess.getoutput(stmt)
    # 执行失败不需要特殊处理，因为该方法无法判断失败成功，只负责将结果进行返回
    return result  # 返回执行结果，但是结果返回的是一个str字符串（不论有多少行）


def writer():
    st.markdown(
        """
        ## 古诗生成 DEMO
        """
    )
    st.sidebar.subheader("配置参数")

    option = st.sidebar.selectbox(
        'Which number do you like best?',
        ['五言绝句', '五言律诗', '七言绝句', '七言律诗'])

    num = st.sidebar.number_input('生成诗歌数量', min_value=1, max_value=10, step=1)

    title_num = 0
    full_length = 65

    if option == '五言绝句':
        title_num = 4
        full_length = 26
    elif option == '五言律诗':
        title_num = 8
        full_length = 50
    elif option == '七言绝句':
        title_num = 4
        full_length = 34
    else:
        title_num = 8
        full_length = 65

    title = st.sidebar.text_input('藏头字', placeholder='填入对应诗歌格式的藏头字数量：{}个字'.format(title_num))

    parser = argparse.ArgumentParser()
    parser.add_argument('--length', default=full_length, type=int, help='生成文本长度')
    parser.add_argument('--batch_size', default=1, type=int, required=False, help='生成的batch size')
    parser.add_argument('--nsamples', default=num, type=int, help='生成样本数量')
    parser.add_argument('--temperature', default=1, type=float, required=False, help='生成温度')
    parser.add_argument('--topk', default=8, type=int, required=False, help='最高几选一')
    parser.add_argument('--topp', default=0, type=float, required=False, help='最高积累概率')
    parser.add_argument('--model_config', default='config/model_config_small.json', type=str, required=False,
                        help='模型参数')
    parser.add_argument('--tokenizer_path', default='vocab/vocab_guwen.txt', type=str, required=False, help='词表路径')
    parser.add_argument('--model_path', default='model/final_model', type=str, required=False, help='模型路径')
    parser.add_argument('--prefix', default=option + '[SEP]' + title, type=str, help='生成文本前缀')
    parser.add_argument('--no_wordpiece', action='store_true', help='不做word piece切词')
    parser.add_argument('--fast_pattern', action='store_true', help='采用更加快的方式生成文本')
    parser.add_argument('--repetition_penalty', default=1.0, type=float, required=False)

    args = parser.parse_args()

    if st.button("点击开始生成"):
        start_message = st.empty()
        start_message.write("正在抽取，请等待...")
        start_time = time.time()
        content = gen_sample(args)
        print(content)
        end_time = time.time()
        start_message.write("抽取完成，耗时{}s".format(end_time - start_time))
        for i in range(num):
            st.text("第{}个结果".format(i + 1))
            st.text(content[i])
    else:
        st.stop()


if __name__ == '__main__':
    writer()