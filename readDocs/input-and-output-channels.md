## 输入和输出通道
https://deepwiki.com/RasaHQ/rasa/6.1-input-and-output-channels

## 文件
- rasa/core/channels/channel.py
- rasa/core/channels/rest.py


## User Message  用户消息
- text: 消息文本内容。



## 输出通道可以发送不同类型的消息：
- send_text_message() ：发送纯文本响应 ， 简单的文本响应
- send_image_url()：发送图片，发送图片、图表等
- send_text_with_buttons()：发送带按钮选项的文本，向用户展示选项
- send_custom_json()：发送平台特定的有效载荷，高级平台功能

## Agent
- rasa/core/agent.py

Agent 是 Rasa 中的核心协调组件。它负责：
- 加载和维护训练好的模型
- 协调消息处理
- 管理对话流程
- 与追踪存储和动作端点交互

代理是处理消息的入口点，并将实际处理委托给消息处理器。
```python
# Agent's main workflow
async def handle_message(self, message: UserMessage) -> Optional[List[Dict[Text, Any]]]:
    processor = self.processor
    await processor.handle_message(message)
```


## Message Processor  消息处理器
- rasa/core/processor.py

消息处理器是一个核心组件，负责处理用户消息的逻辑。它的职责包括：
- 使用 NLU 解析消息
- 更新对话状态跟踪器
- 运行策略以预测下一个动作
- 执行动作
- 处理会话管理


## Actions  动作
- rasa/core/actions/action.py

动作定义了机器人能做什么。主要动作类型包括：
- ActionBotResponse: 用于发送预定义回复
- ActionEndToEndResponse: 用于直接发送生成的响应
- 远程操作：用于调用在单独的动作服务器中实现的动作
- 表单操作：用于处理多轮表单填写


