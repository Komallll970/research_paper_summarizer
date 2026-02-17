<h1 align="center">📘 Research Paper Summarizer</h1>
<h3 align="center"><i>AI-Powered Research Understanding Tool</i></h3>

<p align="center">
An interactive Generative AI application built using <b>LangChain</b>, 
<b>Hugging Face (Zephyr-7B)</b>, and <b>Streamlit</b> to generate 
structured summaries of AI/ML research papers.
</p>

<hr>

<h2>✨ Overview</h2>

<ul>
  <li>📚 Select from influential AI research papers</li>
  <li>🎯 Customize explanation style</li>
  <li>📏 Control summary length</li>
  <li>🧠 Generate structured, context-aware summaries</li>
  <li>🚫 Reduce hallucination with controlled prompting</li>
</ul>

<hr>

<h2>🚀 Core Features</h2>

<ul>
  <li><b>Dynamic Prompt Injection</b> using <code>PromptTemplate</code></li>
  <li><b>Runnable Chain Execution</b> (<code>template | model</code>)</li>
  <li>Multiple explanation styles:
    <ul>
      <li>Beginner-Friendly</li>
      <li>Technical</li>
      <li>Code-Oriented</li>
      <li>Mathematical</li>
    </ul>
  </li>
  <li>Adjustable summary length (Short / Medium / Long)</li>
  <li>Hugging Face API-based inference</li>
  <li>Clean and interactive Streamlit UI</li>
</ul>

<hr>

<h2>🏗️ Architecture</h2>

<pre>
User Input (Streamlit UI)
        ↓
PromptTemplate (Dynamic Formatting)
        ↓
Hugging Face LLM (Zephyr-7B)
        ↓
Structured Summary Output
</pre>

<hr>

<h2>🛠️ Tech Stack</h2>

<ul>
  <li>🐍 Python</li>
  <li>🔗 LangChain</li>
  <li>🤖 Hugging Face Inference API</li>
  <li>🌐 Streamlit</li>
  <li>🧠 Prompt Engineering</li>
</ul>

<hr>

<h2>⚙️ Installation</h2>

<h4>1️⃣ Clone the Repository</h4>

<pre>
git clone https://github.com/your-username/research-paper-summarizer.git
cd research-paper-summarizer
</pre>

<h4>2️⃣ Create Virtual Environment</h4>

<pre>
python -m venv venv
venv\Scripts\activate
</pre>

<h4>3️⃣ Install Dependencies</h4>

<pre>
pip install -r requirements.txt
</pre>

<h4>4️⃣ Add Hugging Face API Token</h4>

Create a <code>.env</code> file:

<pre>
HUGGINGFACEHUB_API_TOKEN=your_token_here
</pre>

<hr>

<h2>▶️ Run the Application</h2>

<pre>
streamlit run app.py
</pre>

Open in browser:
<pre>
http://localhost:8501
</pre>

<hr>

<h2>🧠 Learning Highlights</h2>

<ul>
  <li>✔ Practical LLM integration</li>
  <li>✔ Structured prompt engineering</li>
  <li>✔ Chain-based AI workflows</li>
  <li>✔ API-driven model inference</li>
  <li>✔ Interactive AI application deployment</li>
</ul>

<hr>

<h2>🔮 Future Enhancements</h2>

<ul>
  <li>📄 PDF upload & summarization</li>
  <li>🔎 arXiv API integration (RAG)</li>
  <li>💬 Conversational research assistant</li>
  <li>☁ Deployment on Streamlit Cloud</li>
</ul>

<hr>

<p align="center">
<b>⭐ If you find this project useful, consider giving it a star!</b>
</p>
