from openai import OpenAI
import re, multiprocessing
import sys
# sys.path.append('/nfs-152/disk5/jiaoxuewei/workspace/rl/ly')
# import sandbox_fusion
import json
from datetime import datetime
import threading
import sys
sys.path.append('/nfs-153/jingyi/RLVR/code_base/env/')
# from llm_reward_if import compute_if

# 金融内容安全拦截
from reward_function.llm_sub_reward_content_safe import compute_score_content_safe
from reward_function.llm_sub_reward_dialogue_state_classification_1 import compute_score_dialogue_state_classification_1
from reward_function.llm_sub_reward_dialogue_state_classification import compute_score_dialogue_state_classification
from reward_function.llm_sub_reward_feedback_attribution import compute_score_feedback_attribution
from reward_function.llm_sub_reward_risk_behavior_prediction import compute_score_risk_behavior_prediction
from reward_function.llm_sub_reward_complaint_type_classification_gen import compute_score_complaint_type_classification_gen
from reward_function.llm_sub_reward_complaint_type_classification_extra import compute_score_complaint_type_classification_extra
from reward_function.llm_sub_reward_push_content_compliance_qc import compute_score_push_content_compliance_qc
from reward_function.llm_sub_reward_credit_talk_recommendation import compute_score_credit_talk_recommendation

import random
client_list = []
for i in range(1):
    client = OpenAI(
        base_url=f'http://10.42.87.78:3000{i}/v1',
        api_key="token",
    )
    client_list.append(client)

lock = threading.Lock()
log_file_name = f'/nfs-152/disk5/jingyi/RLVR/reward_logs/dxmbot_{datetime.now().strftime("%Y%m%d_%H%M")}.jsonl'

def log(data, log_file_name):
    with lock:
        with open(log_file_name, 'a') as f:
            f.write(json.dumps(data, ensure_ascii=False)+'\n')
            f.flush()

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    # if "DXM" in data_source:
    #     return compute_score_dxmbot_2(data_source, solution_str, ground_truth, extra_info)
    # elif "企业微信" in data_source:
    #     return compute_score_qiwei(data_source, solution_str, ground_truth, extra_info)
    if "risk" in data_source:
        return compute_score_risk(data_source, solution_str, ground_truth, extra_info)
    elif "dianxiao" in data_source:
        return compute_score_dianxiao(data_source, solution_str, ground_truth, extra_info)
    elif "cuishou" in data_source:
        return compute_score_cuishou(data_source, solution_str, ground_truth, extra_info)
    elif "企业微信" in data_source:
        return compute_score_qiwei(data_source, solution_str, ground_truth, extra_info)
    elif "金融内容安全拦截" in data_source:
        return compute_score_content_safe(data_source, solution_str, ground_truth, extra_info)
    elif "客户对话状态判断_1" in data_source:
        return compute_score_dialogue_state_classification_1(data_source, solution_str, ground_truth, extra_info)
    elif "客户对话状态判断_0" in data_source:
        return compute_score_dialogue_state_classification(data_source, solution_str, ground_truth, extra_info)
    elif "客户反馈归因分析" in data_source:
        return compute_score_feedback_attribution(data_source, solution_str, ground_truth, extra_info)
    elif "客户风险行为预测" in data_source:
        return compute_score_risk_behavior_prediction(data_source, solution_str, ground_truth, extra_info)
    elif "客户投诉类型判断_生成" in data_source:
        return compute_score_complaint_type_classification_gen(data_source, solution_str, ground_truth, extra_info)
    elif "客户投诉类型判断_抽取" in data_source:
        return compute_score_complaint_type_classification_extra(data_source, solution_str, ground_truth, extra_info)
    elif "推送内容合规" in data_source:
        return compute_score_push_content_compliance_qc(data_source, solution_str, ground_truth, extra_info)
    elif "增信话术推荐" in data_source or ("状态判断" in data_source and "客户对话状态判断" not in data_source):
        return compute_score_credit_talk_recommendation(data_source, solution_str, ground_truth, extra_info)
    else:
        raise ValueError("类型错误，没有该类别的奖励！")



def compute_score_dxmbot_2(data_source, solution_str, ground_truth, extra_info=None):
    src_solution_str = solution_str
    # 解析字节think tag
    if '</seed:think>' in solution_str:
        tag = "</seed:think>"
        index = solution_str.rfind(tag)
        solution_str = solution_str[index + len(tag):].strip()  
    elif '</think>' in solution_str:
        tag = "</think>"
        index = solution_str.rfind(tag)
        solution_str = solution_str[index + len(tag):].strip()  
    else:
        return {
            "score": 0.0,
            "acc": False,
            "pred": "",
        }
    # 解析 md json
    if '''```json''' in solution_str:
        # 使用正则表达式提取json内容
        match = re.search(r'```json\n(.*?)\n```', solution_str, re.DOTALL)

        if match:
            extracted_json = match.group(1).strip()  # 提取并去除多余空格
            solution_str = extracted_json
        else:
            print("==========> Error! 未找到json内容",solution_str) 
    else:
        return {
            "score": 0.0,
            "acc": False,
            "pred": "",
        }
    try:
        # "请先输出一段简要的理由说明为何你选择0或1，然后在最后一行请严格只输出一个字符（0 或 1），不要添加其他内容。\n"
        prompt = (
            "请从以下内容中提取 JSON 格式的数据：{response}。\n"
            "然后将提取的 JSON 与参考答案 {ground_truth} 对比，判断语义是否一致，例如：类别编号:类目1与类别编号:1为相同语义。\n"
            "判断标准：只关注核心事件和主要语义，忽略额外的背景信息或前置说明。\n"
            "输出 0 或 1：1 表示一致，0 表示不一致。\n"
            "请严格只输出一个字符（0 或 1），不要添加其他内容。\n"
            "/no_think"
        ).format(response=solution_str, ground_truth=ground_truth)
        
        
        client = random.choice(client_list)
        response = client.chat.completions.create(
            model="general_verifier",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )
        content = choice = response.choices[0].message.content
        # print("==========>", content)
        score = 0.0
        if len(content) > 0:
            try:
                conclusion = float(content.strip().split("\n")[-1].strip())
            except Exception as e:
                conclusion = 0
        else:
            conclusion = 0
        
        correct = conclusion > 0.5
        reward = conclusion
        acc = correct

        log_data = {
            "data_source": data_source,
            "solution_str": src_solution_str,
            "ground_truth": ground_truth,
            "extra_info": extra_info,
            "correct": correct,
            "rm_response":content.strip()
        }
        log(log_data, log_file_name)
        return {
            "score": reward,
            "acc": acc,
            "pred": "",
        }
    except Exception as e:
        print(e)
        log_data = {
            "data_source": data_source,
            "solution_str": src_solution_str,
            "ground_truth": ground_truth,
            "extra_info": extra_info,
            "correct": False
        }
        #log(log_data, math_log_file_name)
        return {
            "score": 0.0,
            "acc": False,
            "pred": "",
        }
        


def compute_score_risk(data_source, solution_str, ground_truth, extra_info=None):
    ground_truth = float(ground_truth)
    src_solution_str = solution_str
    # 解析字节think tag
    if '</seed:think>' in solution_str:
        tag = "</seed:think>"
        index = solution_str.rfind(tag)
        solution_str = solution_str[index + len(tag):].strip()  
    elif '</think>' in solution_str:
        tag = "</think>"
        index = solution_str.rfind(tag)
        solution_str = solution_str[index + len(tag):].strip()  
    else:
        print("=========> 风控", "no think!")
        return {
            "score": -1,
            "acc": False,
            "pred": "",
        }
    # 解析 md json
    if '''```json''' in solution_str:
        # 使用正则表达式提取json内容
        # match = re.search(r'```json(.*?)```', solution_str, re.DOTALL)
        match = re.search(r'```json(.*?)(?=```|$)', solution_str, re.DOTALL)

        if match:
            extracted_json = match.group(1).strip()  # 提取并去除多余空格
            solution_str = extracted_json
        else:
            print("==========> 风控 Error! 未找到json内容",solution_str) 
    else:
        return {
            "score": -1,
            "acc": False,
            "pred": "",
        }
    
    # 比对gt
    try:
        json_result = json.loads(solution_str.replace("\n",""))
        # print("==========>", json_result)
        if json_result["结论"] == "批准放款" and float(ground_truth) == 0:
            correct = True
            acc = True
            reward = 1.5
        elif json_result["结论"] == "拒绝放款" and float(ground_truth) == 1:
            correct = True
            acc = True
            reward = 1.5
        else:
            correct = False
            acc = False
            reward = 0
        log_data = {
            "data_source": data_source,
            "solution_str": src_solution_str,
            "ground_truth": ground_truth,
            "extra_info": extra_info,
            "correct": correct,
            "rm_response":solution_str
        }
        log(log_data, log_file_name)
        return {
            "score": reward,
            "acc": acc,
            "pred": "",
        }
    except Exception as e:
        print("==========> 风控", e, solution_str.replace("\n",""))
        log_data = {
            "data_source": data_source,
            "solution_str": src_solution_str,
            "ground_truth": ground_truth,
            "extra_info": extra_info,
            "correct": False
        }
        #log(log_data, math_log_file_name)
        return {
            "score": -1,
            "acc": False,
            "pred": "",
        }

def compute_score_dianxiao(data_source, solution_str, ground_truth, extra_info=None):
    ground_truth = ground_truth
    src_solution_str = solution_str
    # 解析字节think tag
    if '</seed:think>' in solution_str:
        tag = "</seed:think>"
        index = solution_str.rfind(tag)
        solution_str = solution_str[index + len(tag):].strip()  
    elif '</think>' in solution_str:
        tag = "</think>"
        index = solution_str.rfind(tag)
        solution_str = solution_str[index + len(tag):].strip()  
    else:
        print("=========> 电销", "no think!")
        return {
            "score": -1,
            "acc": False,
            "pred": "",
        }
    # 解析 md json
    if '''```json''' in solution_str:
        # 使用正则表达式提取json内容
        # match = re.search(r'```json(.*?)```', solution_str, re.DOTALL)
        match = re.search(r'```json(.*?)(?=```|$)', solution_str, re.DOTALL)

        if match:
            extracted_json = match.group(1).strip()  # 提取并去除多余空格
            solution_str = extracted_json
        else:
            print("==========> 电销 Error! 未找到json内容",solution_str) 
    else:
        return {
            "score": -1,
            "acc": False,
            "pred": "",
        }
    
    # 比对gt
    try:
        json_result = json.loads(solution_str.replace("\n",""))
        # print("==========>", json_result)
        if json_result["用户情绪状态"] == "用户骂人" and ground_truth == "用户骂人":
            correct = True
            acc = True
            reward = 1.0
        elif json_result["用户情绪状态"] == "用户拒绝客服打电话" and ground_truth == "用户拒绝客服打电话":
            correct = True
            acc = True
            reward = 1.0
        elif json_result["用户情绪状态"] == "其他" and ground_truth == "其他":
            correct = True
            acc = True
            reward = 1.0
        else:
            correct = False
            acc = False
            reward = 0.0
        log_data = {
            "data_source": data_source,
            "solution_str": src_solution_str,
            "ground_truth": ground_truth,
            "extra_info": extra_info,
            "correct": correct,
            "rm_response":solution_str
        }
        log(log_data, log_file_name)
        return {
            "score": reward,
            "acc": acc,
            "pred": "",
        }
    except Exception as e:
        print("==========> 电销 ", e, solution_str.replace("\n",""))
        log_data = {
            "data_source": data_source,
            "solution_str": src_solution_str,
            "ground_truth": ground_truth,
            "extra_info": extra_info,
            "correct": False
        }
        #log(log_data, math_log_file_name)
        return {
            "score": -1,
            "acc": False,
            "pred": "",
        }

def compute_score_qiwei(data_source, solution_str, ground_truth, extra_info=None):
    try:
        ground_truth = ground_truth.replace("```json","").replace("```","").replace("\n", "")
        ground_truth = json.loads(ground_truth)["router"]
    except Exception as e:
        print("==========> 企微 gt error! ", e, ground_truth)
        return {
            "score": 0,
            "acc": False,
            "pred": "",
        }
    src_solution_str = solution_str
    # 解析字节think tag
    if '</seed:think>' in solution_str:
        tag = "</seed:think>"
        index = solution_str.rfind(tag)
        solution_str = solution_str[index + len(tag):].strip()  
    elif '</think>' in solution_str:
        tag = "</think>"
        index = solution_str.rfind(tag)
        solution_str = solution_str[index + len(tag):].strip()  
    else:
        print("=========> 企微", "no think!")
        return {
            "score": -1,
            "acc": False,
            "pred": "",
        }
    # 解析 md json
    if '''```json''' in solution_str:
        # 使用正则表达式提取json内容
        # match = re.search(r'```json(.*?)```', solution_str, re.DOTALL)
        match = re.search(r'```json(.*?)(?=```|$)', solution_str, re.DOTALL)

        if match:
            extracted_json = match.group(1).strip()  # 提取并去除多余空格
            solution_str = extracted_json
        else:
            print("==========> 企微 Error! 未找到json内容",solution_str) 
    else:
        return {
            "score": -1,
            "acc": False,
            "pred": "",
        }
    
    # 比对gt
    try:
        json_result = json.loads(solution_str.replace("\n",""))
        # print("==========>", json_result)
        if json_result["router"] == ground_truth:
            correct = True
            acc = True
            reward = 1.5
        else:
            correct = False
            acc = False
            reward = 0.0
        log_data = {
            "data_source": data_source,
            "solution_str": src_solution_str,
            "ground_truth": ground_truth,
            "extra_info": extra_info,
            "correct": correct,
            "rm_response":solution_str
        }
        log(log_data, log_file_name)
        return {
            "score": reward,
            "acc": acc,
            "pred": "",
        }
    except Exception as e:
        print("==========> 企微", e, solution_str.replace("\n",""))
        log_data = {
            "data_source": data_source,
            "solution_str": src_solution_str,
            "ground_truth": ground_truth,
            "extra_info": extra_info,
            "correct": False
        }
        #log(log_data, math_log_file_name)
        return {
            "score": -1,
            "acc": False,
            "pred": "",
        }

def extract_labels(text):
    """
    从违规内容中提取所有冒号前的标签，如：
    '1、不当话术：威胁联系家人；2、虚构场景：谎称已取消分期资格。'
    返回 ['不当话术', '虚构场景']
    """
    # 提取所有 "数字、可选标点+标签名：" 形式的标签
    return re.findall(r'\d+[、\.]?\s*([^：:]+)：', text)

def compute_score_cuishou(data_source, solution_str, ground_truth, extra_info=None):
    ground_truth = json.loads(ground_truth)
    src_solution_str = solution_str
    # 解析字节think tag
    if '</seed:think>' in solution_str:
        tag = "</seed:think>"
        index = solution_str.rfind(tag)
        solution_str = solution_str[index + len(tag):].strip()  
    elif '</think>' in solution_str:
        tag = "</think>"
        index = solution_str.rfind(tag)
        solution_str = solution_str[index + len(tag):].strip()  
    else:
        print("=========> 催收", "no think!")
        return {
            "score": -1,
            "acc": False,
            "pred": "",
        }
    # 解析 md json
    if '''```json''' in solution_str:
        # 使用正则表达式提取json内容
        # match = re.search(r'```json(.*?)```', solution_str, re.DOTALL)
        match = re.search(r'```json(.*?)(?=```|$)', solution_str, re.DOTALL)

        if match:
            extracted_json = match.group(1).strip()  # 提取并去除多余空格
            solution_str = extracted_json
        else:
            print("==========> 催收 Error! 未找到json内容",solution_str) 
    else:
        return {
            "score": -1,
            "acc": False,
            "pred": "",
        }
    
    # 比对gt
    try:
        json_result = json.loads(solution_str.replace("\n",""))
        # print("==========>", json_result)
        if json_result["坐席是否违规"] == ground_truth["坐席是否违规"] and json_result["违规内容"] == ground_truth["违规内容"]:
            correct = True
            acc = True
            reward = 1.0
        elif json_result["坐席是否违规"] == ground_truth["坐席是否违规"]:
            model_tag = set(extract_labels(json_result["违规内容"]))
            answer_tag = set(extract_labels(ground_truth["违规内容"]))
            if model_tag == answer_tag:
                correct = True
                acc = True
                reward = 1.0
            else:
                correct = False
                acc = False
                reward = 0.0
        else:
            correct = False
            acc = False
            reward = 0.0
        log_data = {
            "data_source": data_source,
            "solution_str": src_solution_str,
            "ground_truth": ground_truth,
            "extra_info": extra_info,
            "correct": correct,
            "rm_response":solution_str
        }
        log(log_data, log_file_name)
        return {
            "score": reward,
            "acc": acc,
            "pred": "",
        }
    except Exception as e:
        print("==========> 催收", e, solution_str.replace("\n",""))
        log_data = {
            "data_source": data_source,
            "solution_str": src_solution_str,
            "ground_truth": ground_truth,
            "extra_info": extra_info,
            "correct": False
        }
        #log(log_data, math_log_file_name)
        return {
            "score": -1,
            "acc": False,
            "pred": "",
        }
        
    



if __name__ == "__main__":
    print(compute_score("cuishou", '''<think>sdasdasd</seed:think>```json
{"坐席是否违规": "否", "违规内容": "无"}
```''', '''{"坐席是否违规": "否", "违规内容": "无"}'''))

