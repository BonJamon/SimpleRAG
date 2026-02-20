class RAG:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    
    async def stream_answer(self, query):
        async for chunk in self.pipeline.stream_answer(query):
            yield f"data: {chunk}\n\n"

            