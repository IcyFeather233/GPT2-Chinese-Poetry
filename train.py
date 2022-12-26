import argparse
import os.path
import random
from datetime import datetime
import torch
import transformers
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from transformers import tokenization_bert


def build_files(data_path: str, tokenized_data_path: str, num_pieces: int,
                full_tokenizer: tokenization_bert.BertTokenizer):
    """
    将训练语料tokenize之后存储
    :param data_path: 原始训练语料的位置
    :param tokenized_data_path: tokenize之后语料的位置
    :param num_pieces: 要将训练语料分成多少份
    :param full_tokenizer: 使用什么tokenizer
    :return:
    """

    # 读取训练原始语料中的诗歌格式和诗句内容
    form, title, content = np.loadtxt(data_path, delimiter=',', skiprows=1, unpack=True, usecols=(0, 4, 5),
                                      encoding='utf-8', dtype=str)
    # 把每行数据变成一行这样的格式：格式[SEP]藏头字[SEP]诗歌内容
    lines = []
    for i in range(len(form)):
        line = form[i] + '[SEP]' + title[i] + '[SEP]' + content[i].replace('\n', '[SEP]')
        lines.append(line)

    # 数据数量
    all_len = len(lines)

    if not os.path.exists(tokenized_data_path):
        os.mkdir(tokenized_data_path)

    for i in tqdm(range(num_pieces)):
        sublines = lines[all_len // num_pieces * i: all_len // num_pieces * (i + 1)]
        # 把尾部例子添加到最后一个piece
        if i == num_pieces - 1:
            sublines.extend(lines[all_len // num_pieces * (i + 1):])
        # tokenizer根据词表切分句子
        sublines = [full_tokenizer.tokenize(line) for line in sublines]

        # 把切分后的字词换成对应的id
        sublines = [full_tokenizer.convert_tokens_to_ids(line) for line in sublines]
        full_line = []
        for subline in sublines:
            # 每行开头添加[MASK]表示开始
            full_line.append(full_tokenizer.convert_tokens_to_ids('[MASK]'))
            full_line.extend(subline)
            # 每行结束添加[CLS]表示结束
            full_line.append(full_tokenizer.convert_tokens_to_ids('[CLS]'))
        # 把切分的第i份内容写入到第i份tokenized了的文件
        with open(tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'w') as f:
            for id in full_line:
                f.write(str(id) + ' ')
        print('finish')




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', default='config/model_config_small.json', type=str, required=False, help='选择模型参数')
    parser.add_argument('--tokenizer_path', default='vocab/vocab_guwen.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--raw_data_path', default='data/train.csv', type=str, required=False, help='选择原始训练数据')
    parser.add_argument('--tokenized_data_path', default='data/tokenized/', type=str, required=False, help='tokenized语料存放位置')
    parser.add_argument('--raw', action='store_true', default=False, help='是否需要先tokenize')
    parser.add_argument('--epochs', default=5, type=int, required=False, help='训练循环次数')
    parser.add_argument('--batch_size', default=8, type=int, required=False, help='训练batch size')
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='学习率')
    parser.add_argument('--stride', default=768, type=int, required=False, help='训练时取训练数据的窗口步长')
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='梯度积累')
    parser.add_argument('--num_pieces', default=100, type=int, required=False, help='将训练语料分成多少份')
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up步数')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--log_step', default=1, type=int, required=False, help='多少步汇报一次loss，必须是gradient accumulation的整数倍')
    parser.add_argument('--output_dir', default='model/', type=str, required=False, help='模型输出路径')
    parser.add_argument('--pretrained_model', default='', type=str, required=False, help='预训练模型')
    parser.add_argument('--writer_dir', default='tensorboard_summary/', type=str, required=False,
                        help='Tensorboard路径')

    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device:', device)

    # 这里用的是transformers最新的gpt2包
    model_config = transformers.modeling_gpt2.GPT2Config.from_json_file(args.model_config)
    print('config:\n' + model_config.to_json_string())

    # 样本的长度限制
    n_ctx = model_config.n_ctx

    # 使用BertTokenizer，选择的词库存放在tokenizer_path
    full_tokenizer = tokenization_bert.BertTokenizer(vocab_file=args.tokenizer_path, tokenize_chinese_chars=True)
    full_tokenizer.max_len = 999999
    raw_data_path = args.raw_data_path
    tokenized_data_path = args.tokenized_data_path
    raw = args.raw
    stride = args.stride
    epochs = args.epochs
    batch_size = args.batch_size
    gradient_accumulation = args.gradient_accumulation
    lr = args.lr
    max_grad_norm = args.max_grad_norm
    num_pieces = args.num_pieces
    warmup_steps = args.warmup_steps
    log_step = args.log_step
    output_dir = args.output_dir
    tb_writer = SummaryWriter(log_dir=args.writer_dir)
    # 检查一下汇报步数是否是gradient accumulation的整数倍
    assert log_step % gradient_accumulation == 0

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # 是否需要先tokenize
    if raw:
        print('正在tokenize训练语料')
        build_files(data_path=raw_data_path, tokenized_data_path=tokenized_data_path, num_pieces=num_pieces,
                    full_tokenizer=full_tokenizer)
        print('训练语料tokenize完毕')

    # 如果设置了预训练模型就读取，如果没有就按照模型初始默认设置
    if not args.pretrained_model:
        model = transformers.modeling_gpt2.GPT2LMHeadModel(config=model_config)
    else:
        model = transformers.modeling_gpt2.GPT2LMHeadModel.from_pretrained(args.pretrained_model)
    # 设置为训练模式
    model.train()
    model.to(device)

    # 打印参数数量
    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    print('number of parameters: {}'.format(num_parameters))

    full_len = 0
    print('计算总共需要的步数')
    for i in tqdm(range(num_pieces)):
        with open(tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'r') as f:
            full_len += len([int(item) for item in f.read().strip().split()])
    total_steps = int(full_len / stride * epochs / batch_size / gradient_accumulation)
    print('total steps = {}'.format(total_steps))

    optimizer = transformers.AdamW(model.parameters(), lr=lr, correct_bias=True)
    scheduler = transformers.WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps,
                                                  t_total=total_steps)
    print('start training')
    overall_step = 0
    running_loss = 0
    for epoch in range(epochs):
        print('epoch {}'.format(epoch + 1))
        now = datetime.now()
        print('time: {}'.format(now))
        # x是从0到numpieces-1的空间中均匀分了numpieces份
        x = np.linspace(0, num_pieces - 1, num_pieces, dtype=np.int32)
        random.shuffle(x)
        piece_num = 0
        # 随机读取tokenized了的训练数据
        for i in x:
            with open(tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'r') as f:
                line = f.read().strip()
            tokens = line.split()
            tokens = [int(token) for token in tokens]
            start_point = 0
            # 切分tokens数据，每份数据的长度为n_ctx，每次移动stride的长度进行切分，切分完毕装入sample列表中
            samples = []
            while start_point < len(tokens) - n_ctx:
                samples.append(tokens[start_point: start_point + n_ctx])
                start_point += stride
            if start_point < len(tokens):
                samples.append(tokens[len(tokens) - n_ctx:])
            random.shuffle(samples)
            # 再把sample分成batch，一个batch有batch_size个sample单位
            for step in range(len(samples) // batch_size):
                # 准备batch数据
                batch = samples[step * batch_size: (step + 1) * batch_size]
                batch_inputs = []
                for ids in batch:
                    int_ids = [int(x) for x in ids]
                    batch_inputs.append(int_ids)
                batch_inputs = torch.tensor(batch_inputs).long().to(device)

                # 前向传播
                outputs = model.forward(input_ids=batch_inputs, labels=batch_inputs)
                loss, logits = outputs[:2]

                # 如果使用了梯度累加
                if gradient_accumulation > 1:
                    loss = loss / gradient_accumulation

                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm(model.parameters(), max_grad_norm)

                if (overall_step + 1) % gradient_accumulation == 0:
                    running_loss += loss.item()
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                if (overall_step + 1) % log_step == 0:
                    tb_writer.add_scalar('loss', loss.item() * gradient_accumulation, overall_step)
                    print('now time: {}:{}. Step {} of piece {} of epoch {}, loss {}'.format(
                        datetime.now().hour,
                        datetime.now().minute,
                        step + 1,
                        piece_num,
                        epoch + 1,
                        running_loss * gradient_accumulation / (log_step / gradient_accumulation)
                    ))
                    running_loss = 0
                overall_step += 1
            piece_num += 1

        print('saving model for epoch {}'.format(epoch + 1))
        if not os.path.exists(output_dir + 'model_epoch{}'.format(epoch + 1)):
            os.mkdir(output_dir + 'model_epoch{}'.format(epoch + 1))
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(output_dir + 'model_epoch{}'.format(epoch + 1))
        print('epoch {} finished'.format(epoch + 1))

        then = datetime.now()
        print('time: {}'.format(then))
        print('time for this one epoch: {}'.format(then - now))

    print('training finished')
    if not os.path.exists(output_dir + 'final_model'):
        os.mkdir(output_dir + 'final_model')
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir + 'final_model')


if __name__ == '__main__':
    main()
