from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st # type: ignore
from langchain_core.prompts import load_prompt
from langchain_core.prompts import PromptTemplate

load_dotenv()

# ✅ Hugging Face Model (API based, not local)
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    temperature=0.5
)

model = ChatHuggingFace(llm=llm)

st.header('Research Tool')

paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis",

        # 🔥 Transformer & NLP Papers
        "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
        "ALBERT: A Lite BERT for Self-supervised Learning",
        "DistilBERT: A Distilled Version of BERT",
        "XLNet: Generalized Autoregressive Pretraining",
        "T5: Exploring the Limits of Transfer Learning",
        "ERNIE: Enhanced Representation through Knowledge Integration",
        "ELECTRA: Pre-training Text Encoders as Discriminators",
        "DeBERTa: Decoding-enhanced BERT",
        "GPT-2: Language Models are Unsupervised Multitask Learners",
        "GPT-4 Technical Report",
        "LLaMA: Open and Efficient Foundation Language Models",
        "PaLM: Scaling Language Modeling with Pathways",
        "Chinchilla: Training Compute-Optimal Large Language Models",
        "Switch Transformers: Scaling to Trillion Parameters",
        "Mixture of Experts Models",
        "UL2: Unifying Language Learning Paradigms",
        "Transformer-XL: Attentive Language Models Beyond Fixed-Length Context",
        "Longformer: The Long-Document Transformer",
        "Reformer: The Efficient Transformer",
        "BigBird: Transformers for Longer Sequences",

        # 🔥 Vision Papers
        "ImageNet Classification with Deep Convolutional Neural Networks",
        "VGG: Very Deep Convolutional Networks for Large-Scale Image Recognition",
        "ResNet: Deep Residual Learning for Image Recognition",
        "DenseNet: Densely Connected Convolutional Networks",
        "Inception: Going Deeper with Convolutions",
        "EfficientNet: Rethinking Model Scaling",
        "Vision Transformer (ViT)",
        "Swin Transformer: Hierarchical Vision Transformer",
        "MobileNet: Efficient Convolutional Neural Networks",
        "YOLO: You Only Look Once",
        "Faster R-CNN",
        "Mask R-CNN",
        "U-Net: Convolutional Networks for Biomedical Image Segmentation",
        "Segment Anything Model (SAM)",
        "CLIP: Connecting Text and Images",
        "DINO: Self-Supervised Vision Transformers",
        "SimCLR: A Simple Framework for Contrastive Learning",
        "MoCo: Momentum Contrast for Unsupervised Learning",
        "ConvNeXt: A ConvNet for the 2020s",
        "NeRF: Neural Radiance Fields",

        # 🔥 Diffusion & Generative Models
        "DDPM: Denoising Diffusion Probabilistic Models",
        "Stable Diffusion",
        "Imagen: Text-to-Image Diffusion Models",
        "DALL-E",
        "StyleGAN",
        "StyleGAN2",
        "CycleGAN",
        "Pix2Pix",
        "VAE: Auto-Encoding Variational Bayes",
        "GAN: Generative Adversarial Nets",

        # 🔥 Reinforcement Learning
        "Deep Q-Network (DQN)",
        "AlphaGo",
        "AlphaZero",
        "PPO: Proximal Policy Optimization",
        "A3C: Asynchronous Advantage Actor-Critic",
        "SAC: Soft Actor-Critic",
        "TD3: Twin Delayed Deep Deterministic Policy Gradient",
        "Dreamer: Learning Behaviors from Pixels",
        "MuZero",
        "IMPALA: Scalable Distributed RL",

        # 🔥 Self-Supervised & Representation Learning
        "BYOL: Bootstrap Your Own Latent",
        "Barlow Twins",
        "MAE: Masked Autoencoders",
        "BEiT: BERT Pre-Training of Image Transformers",
        "SimSiam",
        "Contrastive Predictive Coding",
        "Word2Vec",
        "GloVe: Global Vectors for Word Representation",
        "FastText",
        "ELMo: Deep Contextualized Word Representations",

        # 🔥 Graph Neural Networks
        "GCN: Semi-Supervised Classification with Graph Convolutional Networks",
        "GraphSAGE",
        "GAT: Graph Attention Networks",
        "DeepWalk",
        "Node2Vec",
        "Graph Transformer Networks",

        # 🔥 Multimodal & LLM Advances
        "Flamingo: Few-Shot Learning for Multimodal Models",
        "BLIP: Bootstrapping Language-Image Pretraining",
        "BLIP-2",
        "Kosmos-1: Multimodal LLM",
        "Gemini Technical Report",
        "Claude Technical Report",
        "Phi: Small Language Models with Big Performance",
        "Mistral 7B",
        "Mixtral: Sparse Mixture of Experts",
        "Falcon: Open Source LLM",

        # 🔥 Optimization & Training
        "Adam: A Method for Stochastic Optimization",
        "Dropout: A Simple Way to Prevent Overfitting",
        "Batch Normalization",
        "Layer Normalization",
        "RMSProp",
        "Learning without Forgetting",
        "Knowledge Distillation",
        "Lottery Ticket Hypothesis",
        "Scaling Laws for Neural Language Models",
        "Attention with Linear Biases (ALiBi)"
    ]
)


style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

length_input = st.selectbox(
    "Select Explanation Length",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)

template = load_prompt('langchain_prompt/template.json') # allow reuability of prompt, we can load it from json file instead of defining it again and again


if st.button('Summarize'):
    chain = template | model
    result = chain.invoke({                  
        'paper_input': paper_input,
        'style_input': style_input,
        'length_input': length_input
    })
    st.write(result.content)


# without chain we have to invoke it 2 times, first to format the prompt and then to get the result from the model
# prompt= template.invoke({           # to fill placeholder        
 #       'paper_input': paper_input,
  #      'style_input': style_input,
   #     'length_input': length_input
    #})
  #  if st.button('Summarize'):
   # result = pmodel.invoke(prompt)
    #st.write(result.content)#