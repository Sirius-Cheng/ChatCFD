import streamlit as st
import PyPDF2
import io
import json
from PIL import Image
import base64
import re
import tiktoken
from datetime import datetime

import config, case_file_requirements, preprocess_OF_tutorial, set_config, main_run_chatcfd, qa_modules
import pathlib
import os
from openai_client_factory import create_chat_client


general_prompt = ''


def _extract_json_dict(text: str):
    """Best-effort JSON parser that tolerates leading/trailing text."""
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


class ChatBot:
    def __init__(self):
        self.client = create_chat_client("DEEPSEEK_R1")
        self.model_name = os.environ.get("DEEPSEEK_R1_MODEL_NAME")
        # self.system_prompt = """You are an intelligent assistant capable of:
        # 1. Maintaining politeness and professionalism
        # 2. Remembering the context of the conversation
        # 3. Processing and analyzing content from documents uploaded by users
        # 4. Answering user questions while keeping the conversation coherent
        #
        # Please always respond in a clear, accurate, and helpful manner."""
        self.system_prompt = """你是一位智能助手，能够：
        1. 保持礼貌与专业
        2. 记住对话上下文
        3. 处理并分析用户上传文档的内容
        4. 在保持对话连贯的前提下回答用户问题

        请始终以清晰、准确且有帮助的方式作答。"""
        self.temperature = 0.9

        self.token_counter = {
            "total": 0,
            "qa_history": []
        }

    def process_pdf(self, pdf_file):
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            # return f"PDF processing error: {str(e)}"
            return f"PDF 处理出错：{str(e)}"

    def get_response(self, messages):

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "system", "content": self.system_prompt}] + messages,
                temperature=self.temperature,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            # Record token usage
            usage = response.usage
            self.token_counter["total"] += usage.total_tokens
            qa_record = {
                "prompt": messages,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "timestamp": datetime.now().isoformat()
            }
            return response.choices[0].message.content
        except Exception as e:
            # return f"Chat error: {str(e)}"
            return f"聊天出错：{str(e)}"

    def count_tokens(self, text: str, model: str = "gpt-4o") -> int:
        """Use tiktoken to count the number of tokens"""
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = ChatBot()
    if "file_content" not in st.session_state:
        st.session_state.file_content = None
    if "file_processed" not in st.session_state:
        st.session_state.file_processed = False
    if "ask_case_solver" not in st.session_state:
        st.session_state.ask_case_solver = False
    if "user_answered" not in st.session_state:
        st.session_state.user_answered = False
    if "user_answer_finished" not in st.session_state:
        st.session_state.user_answer_finished = False
    if "uploaded_grid" not in st.session_state:
        st.session_state.uploaded_grid = False
    if "show_start" not in st.session_state:
        st.session_state.show_start = False

def extract_pure_response(text):
    # Use regex to match all content (including newlines)
    pattern = r"Here is my response:(.*?)(?=$|\Z)"
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        # Remove leading and trailing whitespace
        return match.group(1).strip()
    return ""

def test_function_call_by_QA():
    """Test function call"""
    # print("the test_function_call_by_QA() is called")  # Console print
    print("test_function_call_by_QA() 测试函数已被调用")  # Console print
    # return "✅ Test function successfully called! System status normal."
    return "✅ 测试函数调用成功！系统状态正常。"
    

def main():

    # test other functions

    # test_function_call_by_QA()

    # a = 1

    # streamlit functions

    # st.title("ChatCFD: chat to run CFD cases.")
    st.title("ChatCFD：通过聊天运行CFD案例。")

    st.divider()
    
    initialize_session_state()

    with st.sidebar:

        # Export chat history functionality
        # st.header("Export chat history")
        st.header("导出聊天记录")
        export_format = "JSON"
        
        # if st.button("Export chat"):
        if st.button("导出对话"):
            if not st.session_state.messages:
                # st.warning("Empty chat history")
                st.warning("聊天记录为空")
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"chatlog_{timestamp}"

                chat_data = {
                    "metadata": {
                        "export_time": datetime.now().isoformat(),
                        "total_messages": len(st.session_state.messages),
                        "total_tokens": st.session_state.chatbot.token_counter["total"]
                    },
                    "messages": st.session_state.messages
                }
                
                # st.sidebar.download_button(
                #     label="Download JSON file",
                #     data=json.dumps(chat_data, indent=2, ensure_ascii=False),
                #     file_name=f"{filename}.json",
                #     mime="application/json"
                # )
                st.sidebar.download_button(
                    label="下载 JSON 文件",
                    data=json.dumps(chat_data, indent=2, ensure_ascii=False),
                    file_name=f"{filename}.json",
                    mime="application/json"
                )

    # Sidebar: File Upload
    with st.sidebar:
        # st.header("Upload the document")
        st.header("上传文档")
        # uploaded_file = st.file_uploader(
        #     "Please upload PDF",
        #     type=['pdf']
        # )
        uploaded_file = st.file_uploader(
            "请上传 PDF",
            type=['pdf']
        )
        
        if uploaded_file:
            if not st.session_state.file_processed:
                if uploaded_file.type == "application/pdf":

                    save_dir = pathlib.Path(config.TEMP_PATH)
                    
                    try:
                        # Build save path
                        file_path = save_dir / uploaded_file.name.replace(" ", "_")
                        
                        # Save uploaded file
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        config.pdf_path =  f"{config.TEMP_PATH}/{uploaded_file.name}"

                    except Exception as e:
                        # st.error(f"Failed at processed the pdf file: {str(e)}") 
                        st.error(f"处理 PDF 文件失败：{str(e)}") 

                    text_content = st.session_state.chatbot.process_pdf(uploaded_file)
                    config.paper_content = text_content
                    # st.session_state.file_content = f"The  contents：\n{text_content}"
                    st.session_state.file_content = f"文档内容：\n{text_content}"
                    # st.toast("PDF uploaded！", icon="💾")
                    st.toast("PDF 已上传！", icon="💾")
                    
                    # Add 1st question
                    question_1 = f'''所附的PDF包含了几个CFD案例，我希望之后能亲自运行其中一个或几个案例。请阅读这篇论文，并列出所有不同的CFD案例及其特征描述。为每个案例分配一个标签，格式为 Case_X（例如 Case_1, Case_2）。请使用中文回答

                    - 请将每个导致独立模拟运行的唯一参数组合计为一个CFD案例。这些参数包括但不限于：几何形状、边界条件、流动参数（雷诺数Re/马赫数Mach/攻角AoA/速度）、物理模型或求解器。
                    - 如果为了统计分析或收敛性研究而对同一组参数进行了多次运行，请将这些计为一个案例，除非论文因其不同目标或条件而将其明确区分为不同案例。
                    - 如果有任何案例是使用OpenFOAM进行模拟的，请识别其所用的求解器，或为其找到一个合适的求解器。在描述案例时，请注明求解器名称。
                    
                    论文内容如下： \n{text_content}. 
                    '''

                    # question_1 = f'''The attached PDF contain several CFD cases, and I would like to run one or several of the case by my self later. Please read the paper and list all distinct CFD cases with characteristic description. Give each case a tag as Case_X (such as Case_1, Case_2).

                    # - Please count each unique combination of parameters that results in a separate simulation run as one CFD case. These parameters include but not limited to the geometry, boundary Conditions, flow Parameters (Re/Mach/AoA/velocity), physical Model, or Solver.
                    # - If there are multiple runs of the same parameters for statistical analysis or convergence studies, count these as one case, unless the paper specifies them as distinct due to different goals or conditions.
                    # - If any case is simulated using OpenFOAM, identify the solver or find a proper solver to run the case. Show the solver name when describing the case.
                    
                    # The paper is as follows: \n{text_content}. 
                    # '''
                    st.session_state.messages.append({
                        "role": "user",
                        "content": question_1, "timestamp": datetime.now().isoformat()
                    })
                    
                    # Get response for question A
                    with st.chat_message("assistant"):
                        response_1 = st.session_state.chatbot.get_response(st.session_state.messages)
                        st.write(response_1)
                        st.session_state.messages.append({"role": "assistant", "content": response_1, "timestamp": datetime.now().isoformat()})

                    st.session_state.file_processed = True

                    # Chatbot ask the user to choose case and solver
                    if not st.session_state.ask_case_solver:
                        # ask_to_choose_case_and_solver = '''Please choose the case you want to simulate and the OpenFOAM solver you want to use. 
                        #     Your answer shall be like one of the followings:\n- I want to simulate the Case with AOA = 10 degree and SpalartAllmaras model.\n- I want to simulate Case_1 using rhoCentralFoam and the SpalartAllmaras model.\n- I want to simulate the Case with AOA = 10 degree and kOmegaSST model.\n
                        #     
                        # \n You must choose only one case.
                        # '''
                        ask_to_choose_case_and_solver = '''请选择你要模拟的案例以及希望使用的 OpenFOAM 求解器。
                            你的回答可以如下：\n- 我想模拟攻角 AOA = 10° 且采用 SpalartAllmaras 模型的案例。\n- 我想模拟 Case_1，并使用 rhoCentralFoam 求解器与 SpalartAllmaras 模型。\n- 我想模拟攻角 AOA = 10° 且采用 kOmegaSST 模型的案例。\n
                            
                        \n 你必须且只能选择一个案例。
                        '''
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": ask_to_choose_case_and_solver,
                            "timestamp": datetime.now().isoformat()
                        })

                        st.session_state.ask_case_solver = True

    with st.sidebar:
        # st.header("Upload the mesh file")
        st.header("上传网格文件")
        # uploaded_mesh_file = st.file_uploader(
        #     "Please upload mesh (only support the Fluent-format .msh)",
        #     type=['msh']
        # )
        uploaded_mesh_file = st.file_uploader(
            "请上传网格文件（仅支持 Fluent 格式 .msh）",
            type=['msh']
        )
        if uploaded_mesh_file:
            if not st.session_state.uploaded_grid:
                # Create save directory
                save_dir = pathlib.Path(config.TEMP_PATH)
                
                try:
                    # Build save path
                    file_path = save_dir / uploaded_mesh_file.name.replace(" ", "_")
                    
                    # Save uploaded file
                    with open(file_path, "wb") as f:
                        f.write(uploaded_mesh_file.getbuffer())
                    
                    # st.toast(f"The mesh file has been saved: {file_path}", icon="💾")
                    st.toast(f"网格文件已保存：{file_path}", icon="💾")

                    config.case_grid = f"{config.TEMP_PATH}/{uploaded_mesh_file.name}"

                    # check the grid using OpenFOAM, later
                    
                    case_file_requirements.extract_boundary_names(file_path)

                    # st.toast(f"The mesh file has been processed! ")
                    st.toast("网格文件处理完成！")

                    boundary_names = ", ".join(config.case_boundaries)

                    config.case_boundary_names = boundary_names

                    # info_after_mesh_processed = f'''You have uploaded a mesh file with boundary names as: {boundary_names}.\nNow the case are prepared and running in the background. Running information will be shown in the console.'''
                    info_after_mesh_processed = f'''你上传的网格文件包含以下边界名称：{boundary_names}。\n案例已准备完毕并在后台运行，运行信息会显示在控制台。'''
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": info_after_mesh_processed,
                        "timestamp": datetime.now().isoformat()
                    })

                    st.session_state.ask_case_solver = True

                    st.session_state.uploaded_grid = True

                except Exception as e:
                    # st.error(f"Failed at processed the mesh file: {str(e)}")              
                    st.error(f"处理网格文件失败：{str(e)}")              

    # Display conversation history
    if len(st.session_state.messages) > 0:
        for message in st.session_state.messages[1:]:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                if message["content"].startswith("Understand the user's answer") or message["content"].startswith("请理解用户的回答"):
                    continue
                else:
                    st.chat_message("assistant").write(message["content"])

    if st.session_state.show_start == False:
        # st.header('**Please upload the paper to start!**')
        st.header('**请上传论文以开始！**')
        st.session_state.show_start = True

    # guide the user to choose cases
    if st.session_state.ask_case_solver == True and st.session_state.user_answered == True:
        a = 1
        try: 
            user_answer = st.chat_messages[-1]['content']
            paper_case_descriptions = st.chat_messages[-1]['content']

            # json_reponse_sample = '''
            # {
            #     "Case_1":{
            #         "solver":"<solver_name>",
            #         "turbulence_model":"<model_name>",
            #         "other_physical_model":"<model_name>",
            #         "case_specific_description":"<specific case discription that differenciate this case from the others in the paper."
            #     },
            #     "Case_2":{
            #         "solver":"<solver_name>",
            #         "turbulence_model":"<model_name>",
            #         "other_physical_model":"<model_name>",
            #         "case_specific_description":"<specific case discription that differenciate this case from the others in the paper."
            #     },
            #     "Case_X":{
            #         "solver":"<solver_name>",
            #         "turbulence_model":"<model_name>",
            #         "other_physical_model":"<model_name>",
            #         "case_specific_description":"<specific case discription that differenciate this case from the others in the paper."
            #     }
            # }
            # '''
            json_reponse_sample = '''
            {
                "Case_1":{
                    "solver":"<求解器名称>",
                    "turbulence_model":"<湍流模型名称>",
                    "other_physical_model":"<其他物理模型名称>",
                    "case_specific_description":"<能够区分该案例与论文中其他案例的特征描述>"
                },
                "Case_2":{
                    "solver":"<求解器名称>",
                    "turbulence_model":"<湍流模型名称>",
                    "other_physical_model":"<其他物理模型名称>",
                    "case_specific_description":"<能够区分该案例与论文中其他案例的特征描述>"
                },
                "Case_X":{
                    "solver":"<求解器名称>",
                    "turbulence_model":"<湍流模型名称>",
                    "other_physical_model":"<其他物理模型名称>",
                    "case_specific_description":"<能够区分该案例与论文中其他案例的特征描述>"
                }
            }
            '''

            # guide_case_choose_prompt = f'''Understand the user's answer and describe the case details of the user's requirement.
            #
            #             The user's answer is:{user_answer}
            #
            #             Please generate JSON content according to these requirements:
            #
            #             1. Strictly follow this example format containing ONLY JSON content:{json_reponse_sample}. For the case_specific_description sections, propose characteristics that can differenciate this case from the other similar cases in the paper. The differentiating characteristics must exclude conventional attributes such as geometry, shape, numerical parameters, physical models, or other standard descriptors. 
            #
            #             2. Absolutely AVOID any non-JSON elements including but not limited to:
            #             - Markdown code block markers (```json or ```)
            #             - Extra comments or explanations
            #             - Unnecessary empty lines or indentation
            #             - Any text outside JSON structure
            #
            #             3. Critical syntax requirements:
            #             - Maintain strict JSON syntax compliance
            #             - Enclose all keys in double quotes
            #             - Use double quotes for string values
            #             - Ensure no trailing comma after last property
            # '''
            guide_case_choose_prompt = f'''请理解用户的回答，并描述其需求对应的案例细节。

                        用户的回答是:{user_answer}

                        请按以下要求生成 JSON 内容：

                        1. 严格遵循仅包含 JSON 的示例格式：{json_reponse_sample}。对于 case_specific_description 字段，请提出能将该案例与论文中其他相似案例区分开的特征，且这些特征不得包含几何、形状、数值参数、物理模型或其他常规描述。

                        2. 严禁出现 JSON 以外的内容，包括但不限于：
                        - Markdown 代码块标记（```json 或 ```）
                        - 额外注释或解释
                        - 不必要的空行或缩进
                        - 任何 JSON 结构之外的文本

                        3. 严格遵守 JSON 语法：
                        - 所有键必须使用双引号
                        - 字符串值必须使用双引号
                        - 最后一个属性后不得出现多余逗号
            '''

            st.chat_message("assistant").write(guide_case_choose_prompt)
            st.session_state.messages.append({"role": "assistant", "content": guide_case_choose_prompt, "timestamp": datetime.now().isoformat()})

            with st.chat_message("assistant"):
                response = st.session_state.chatbot.get_response(st.session_state.messages)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response, "timestamp": datetime.now().isoformat()})

            # prompt_2 = f'''Task: The user want to simulate a CFD case with the following characteristicis,
            # identify the CFD case from the following case descriptions from a PDF.
            # - Characteristics: {user_answer}.
            # - Case descriptions: {paper_case_descriptions}.
            # Your response shall only include the answer without any thinking content.
            # '''
            prompt_2 = f'''任务：用户希望模拟具备以下特征的 CFD 案例，请从 PDF 中的案例描述里识别该案例。
            - 案例特征：{user_answer}.
            - 案例描述：{paper_case_descriptions}.
            仅输出答案，不得包含思考过程。
            '''

        except Exception as e:
            # return f"Chat error: {str(e)}"
            return f"聊天出错：{str(e)}"

    # User input
    # if prompt := st.chat_input("Enter your requirement or reply."):
    if prompt := st.chat_input("请输入您的需求或回复。"):
        
        st.chat_message("user").write(prompt)  # Display the user's original prompt in the UI

        if st.session_state.ask_case_solver and not st.session_state.user_answer_finished: # ask the user for Case_X, solver and turbulence
            # json_reponse_sample = '''
            # {
            #     "Case_1":{
            #         "case_name" = <some_case_name>,
            #         "solver":"<solver_name>",
            #         "turbulence_model":"<model_name>",
            #         "other_physical_model":"<model_name>",
            #         "case_specific_description":"<a sentence that describes the case setup with detailed parameters that differenciate this case from the other cases in the paper>"
            #     }
            # }
            # '''
            json_reponse_sample = '''
            {
                "Case_1":{
                    "case_name" = <案例名称>,
                    "solver":"<求解器名称>",
                    "turbulence_model":"<湍流模型名称>",
                    "other_physical_model":"<其他物理模型名称>",
                    "case_specific_description":"<一段能够通过详细参数区分该案例的描述>"
                }
            }
            '''

            # guide_case_choose_prompt = f'''Understand the user's answer and describe the case details of the user's requirement.
            #
            #             The user's answer is:{prompt}
            #
            #             Please generate JSON content according to these requirements:
            #
            #             1. Strictly follow this example format containing ONLY JSON content:{json_reponse_sample}
            #
            #             2. Absolutely AVOID any non-JSON elements including but not limited to:
            #             - Markdown code block markers (```json or ```)
            #             - Extra comments or explanations
            #             - Unnecessary empty lines or indentation
            #             - Any text outside JSON structure
            #
            #             3. Critical syntax requirements:
            #             - Maintain strict JSON syntax compliance
            #             - Enclose all keys in double quotes
            #             - Use double quotes for string values
            #             - Ensure no trailing comma after last property
            #
            #             4. Case_name must adhere to the following format:
            #              [a-zA-Z0-9_]+ - only containing lowercase letters, uppercase letters, numbers, or underscores. Special characters (e.g. -, @, #, spaces) are not permitted.
            #
            #             5. The solver must be one of the followings: {config.string_of_solver_keywords}. 
            #             The turbulence _model must be one of the followings: {config.string_of_turbulence_model}.
            # '''
            guide_case_choose_prompt = f'''请理解用户的回答，并描述其需求对应的案例细节。

                        用户的回答是:{prompt}

                        请按以下要求生成 JSON 内容：

                        1. 严格遵循仅包含 JSON 的示例格式：{json_reponse_sample}

                        2. 严禁出现 JSON 以外的内容，包括但不限于：
                        - Markdown 代码块标记（```json 或 ```）
                        - 额外注释或解释
                        - 不必要的空行或缩进
                        - 任何 JSON 结构之外的文本

                        3. 严格遵守 JSON 语法：
                        - 所有键必须使用双引号
                        - 字符串值必须使用双引号
                        - 最后一个属性后不得出现多余逗号

                        4. case_name 必须满足格式 [a-zA-Z0-9_]+，只允许字母、数字或下划线，禁止使用特殊字符（如 -, @, #、空格等）。

                        5. Solver 必须从以下选项中选择：{config.string_of_solver_keywords}。
                        湍流模型必须从以下选项中选择：{config.string_of_turbulence_model}。
            '''

            st.session_state.messages.append({"role": "user", "content": guide_case_choose_prompt, "timestamp": datetime.now().isoformat()})

            # Get assistant's response
            with st.chat_message("assistant"):
                response = st.session_state.chatbot.get_response(st.session_state.messages)
                parsed_case_dict = _extract_json_dict(response)
                if not parsed_case_dict:
                    st.error("助手返回的内容不是有效的 JSON，请重试或调整描述。")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "timestamp": datetime.now().isoformat()
                    })
                    return
                config.all_case_dict = parsed_case_dict

                qa = qa_modules.QA_NoContext_deepseek_R1()

                # convert_json_to_md = f'''Convert the provided JSON string into a Markdown format where:
                #     1. Each top-level JSON key becomes a main heading (#)
                #     2. Its corresponding key-value pairs are rendered as unordered list items
                #     3. Maintain the original key-value hierarchy in list format
                #
                #     The provided json string is as follow:{response}.
                # '''
                convert_json_to_md = f'''请将以下 JSON 字符串转换为 Markdown：
                    1. 每个顶层 JSON 键作为一级标题（#）
                    2. 其对应的键值对以无序列表展示
                    3. 保持原有的层级结构

                    需要转换的 JSON 字符串如下：{response}.
                '''

                md_form = qa.ask(convert_json_to_md)

                # decorated_response = f'''You choose to simulate the cases with the following setups:\n{md_form}'''
                decorated_response = f'''你选择模拟的案例配置如下：\n{md_form}'''
                st.write(decorated_response)
                st.session_state.messages.append({"role": "assistant", "content": decorated_response, "timestamp": datetime.now().isoformat()})
                # later, fnae
                st.session_state.user_answer_finished = True

                

        else:   # normal case
            st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": datetime.now().isoformat()})
            # Get assistant's response
            with st.chat_message("assistant"):
                response = st.session_state.chatbot.get_response(st.session_state.messages)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response, "timestamp": datetime.now().isoformat()})

    if st.session_state.file_processed and st.session_state.user_answer_finished and not st.session_state.uploaded_grid:
        # st.write("If you don't have further requirement on the case setup. \n**Please upload the mesh of the Fluent .msh format.**")
        st.write("如果你对案例设置没有更多要求。\n**请上传 Fluent .msh 格式的网格。**")

    if st.session_state.uploaded_grid and st.session_state.file_processed and st.session_state.user_answer_finished:
        # read in preprocess OF tutorials
        # print(f"**************** Preprocessing OF tutorials at {config.of_tutorial_dir} ****************")
        print(f"**************** 在 {config.of_tutorial_dir} 预处理 OF 教程 ****************")
        # if not config.flag_OF_tutorial_processed:
        #     preprocess_OF_tutorial.main()
        #     config.flag_OF_tutorial_processed = True
        preprocess_OF_tutorial.read_in_processed_merged_OF_cases()
        for key, value in config.all_case_dict.items():
            case_name = value["case_name"]
            # print(f"***** start processing {key}: {case_name} *****")
            print(f"***** 开始处理 {key}: {case_name} *****")
            solver = value["solver"]
            turbulence_model = value["turbulence_model"]

            case_specific_description = value["case_specific_description"]

            main_run_chatcfd.test_solver = solver

            main_run_chatcfd.test_turbulence_model = turbulence_model

            main_run_chatcfd.test_case_name = case_name

            main_run_chatcfd.test_case_description = case_specific_description

            main_run_chatcfd.run_case()

            # single_case_builder_runner.single_case_details_from_PDF(case_name, solver, turbulence_model, 
            #     transient, simulation_duration, case_specific_description)

if __name__ == "__main__":
    set_config.read_in_config()
    # set_config.load_openfoam_environment()
    main()