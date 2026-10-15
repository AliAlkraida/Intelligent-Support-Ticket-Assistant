from src.knowledge_based import KnowledgeBasedAssistant

def build_assistant(name):
    if name == "knowledge_based":
        return KnowledgeBasedAssistant()
    if name == "rag":
        from src.rag import RAGAssistant
        return RAGAssistant()
    if name == "llm_prompting":
        from src.llm_prompting import PromptingAssistant
        return PromptingAssistant()
    raise ValueError(f"Unknown assistant: {name}")
