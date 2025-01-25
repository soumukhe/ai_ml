# DeepSeek Platform: https://platform.deepseek.com/

# Docs: https://api-docs.deepseek.com/

# DeepSeek Models: DeepSeek-chat and DeepSeek-reasoner (DeepSeek-R1)

DeepSeek-chat and DeepSeek-reasoner are two different models developed by DeepSeek, each designed for specific tasks and use cases. Below are their features and key differences:

---

## **DeepSeek-chat**
Also referred to as **DeepSeek-V3**, this is a general-purpose language model designed for a wide range of conversational tasks.

### **Key Features**
- **Open-source**: 671B parameter Mixture-of-Experts (MoE) model with 37B activated parameters per token.
- **Context Window**: 128K token context window.
- **Training Data**: Trained on 14.8T tokens.
- **Multi-token Prediction**: Supports multi-token prediction and incorporates innovative load balancing.
- **Pricing**: More affordable compared to DeepSeek-R1:
  - **Input Cost**: $0.14 per million tokens.
  - **Output Cost**: $0.28 per million tokens.

---

## **DeepSeek-reasoner (DeepSeek-R1)**
Also known as **DeepSeek-R1**, this is a specialized model focused on **advanced reasoning capabilities**.

### **Key Features**
- **Open-source**: 671B parameter Mixture-of-Experts (MoE) model with 37B activated parameters per token.
- **Reinforcement Learning**:
  - Trained via large-scale reinforcement learning with a focus on reasoning capabilities.
  - Includes two RL stages to discover improved reasoning patterns and align with human preferences.
- **Supervised Fine-Tuning (SFT)**:
  - Two SFT stages to seed reasoning and non-reasoning capabilities.
- **Pricing**: Higher cost compared to DeepSeek-V3:
  - **Input Cost**: $0.55 per million tokens.
  - **Output Cost**: $2.19 per million tokens.

---

## **Key Differences**
1. **Focus**:  
   - **DeepSeek-chat**: General-purpose conversational model.  
   - **DeepSeek-reasoner**: Specializes in advanced reasoning tasks.

2. **Training Approach**:  
   - **DeepSeek-reasoner**: Utilizes reinforcement learning for reasoning patterns.  
   - **DeepSeek-chat**: Relies on traditional training methods.

3. **Reasoning Capabilities**:  
   - **DeepSeek-reasoner**: Excels in complex reasoning, problem-solving, and logical deduction.  
   - **DeepSeek-chat**: Geared toward versatile and general conversational use.

4. **Pricing**:  
   - **DeepSeek-reasoner**: More expensive.  
   - **DeepSeek-chat**: Cost-effective for general use.

5. **Performance**:  
   - **DeepSeek-reasoner**: Comparable to OpenAI's o1 series in math, code, and reasoning tasks.

---

## **Summary**
- **DeepSeek-chat**: A versatile, cost-effective solution for general conversational tasks.  
- **DeepSeek-reasoner**: A powerful tool for specialized reasoning and problem-solving.

---

## **Model Names for Inferencing**
- For **DeepSeek-chat**:  
  `model="deepseek-chat"`

- For **DeepSeek-reasoner**:  
  `model="deepseek-reasoner"`
