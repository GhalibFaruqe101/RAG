import gradio as gr
import RAG_sys


def gradio_chat(message, history):
    if message.strip().lower() in ["/stop", "stop"]:
        import os
        import signal

        os.kill(os.getpid(), signal.SIGINT)
        return "Shutting down the server..."

    try:
        result = RAG_sys.ask(message)

        if not isinstance(result, str):
            result = str(result)

        return result
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        return error_msg


demo = gr.ChatInterface(
    fn=gradio_chat,
    title="RAG Document Chatbot",
    description=" Type '/stop' to gracefully shut down the server.",
    examples=[
        "Which university does Hunter Jacobson studies mentioned in the resume? "
        "He is a student of human resource"
    ],
    stop_btn=True,
)

if __name__ == "__main__":
    demo.launch(share=False, server_port=7861, debug=True)
