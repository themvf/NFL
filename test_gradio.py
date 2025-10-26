import gradio as gr

def test_function(name):
    return f"Hello {name}!"

with gr.Blocks() as app:
    gr.Markdown("# Test Gradio App")
    name_input = gr.Textbox(label="Name", value="World")
    output = gr.Textbox(label="Output")
    btn = gr.Button("Test")
    btn.click(test_function, inputs=[name_input], outputs=[output])

if __name__ == "__main__":
    print("Starting Gradio app...")
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)
