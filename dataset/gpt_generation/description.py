import json
from openai import OpenAI
from tqdm import tqdm
import os.path as osp



global_api_key = "sk-PdEqIkKM9Nu6EnkJ101672DaE0954d95A7F64eF62083Af01" # Enter Your API Key!!!
#class description缓存
descriptions = None

templates = ["What does {} look like among all {}? ",
             "What are the distinct features of {} for recognition among all {}? ",
             "How can you identify {} in appearance among all {}? ",
             "What are the differences between {} and other {} in appearance? ",
             "What visual cue is unique to {} among all {}? "]

infos = {
    'ImageNet':             ["{}",                "objects"],
    'cifar100':             ["{}",                "objects"],
    'OxfordPets':           ["a pet {}",          "types of pets"],
}

def get_completion(client, prompt, model="gpt-3.5-turbo", temperature=1):
    """
    调用openai的api接口，给出text提示，返回description
    Args:
        client:
        prompt:
        model:
        temperature:
    Returns:
    """
    messages = [{"role": "system", "content": "You are good at image classification."}, {"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()

def get_All_Descriptions(args):
    # 要使用全局变量并赋值，需要使用global关键字
    global descriptions
    # 1. 若没有缓存
    if descriptions is None:
        path = osp.join(args.gpt_dir ,'description', args.db_name + '.json')
        # 1.1 若文件存在，则直接读取
        if osp.isfile(path):
            with open(path, 'r') as f:
                descriptions = json.load(f)
        # 1.2 若文件不存在，则调用gpt生成description
        else:
            # 1.2.1 ——1.0.0版本之后的openai接口调用代码
            client = OpenAI(
                api_key=global_api_key,
                base_url="https://free.gpt.ge/v1/"
            )
            # 1.2.2 先读取该数据集中所有的class
            cls_names_path = osp.join(args.gpt_dir , 'classname' , args.db_name + '.txt')
            with open(cls_names_path, 'r') as f:
                classnames = f.read().split("\n")[:-1]

            descriptions = {}
            cls_descrip_path = osp.join(args.gpt_dir, 'description', args.db_name + '.json')
            # 1.2.3 对于所有class，逐个生成description
            for classname in tqdm(classnames):
                info = infos[args.db_name]
                # prompt替换和生成
                prompts = [template.format(info[0], info[1]).format(classname) + "Describe it in 20 words." for template
                           in templates]
                # print("\r\n prompts:{}".format(prompts))
                responses = [get_completion(client, prompt) for prompt in prompts]
                # 放入缓存中
                descriptions[classname] = responses

            # 1.2.4 将缓存内容写入json文件——indent，根据数据格式缩进显示，读起来更加清晰
            with open(cls_descrip_path, 'w') as f:
                json.dump(descriptions, f, indent=4)
    return descriptions

def get_Classes_Descriptions(args, classnames):
    descriptions = get_All_Descriptions(args)
    # 2. 从缓存中读取对应classes的descriptions
    reuslt=[descriptions[i] for i in classnames]
    return reuslt


# from main_incremental_submit import parse_option
# if __name__ == "__main__":
#     args = parse_option()
#     get_All_Descriptions(args)
    # get_Classes_Descriptions(args,["face", "leopard", "motorbike"])