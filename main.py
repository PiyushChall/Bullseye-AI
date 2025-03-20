from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import asyncio
import os

# Configure Gemini API (replace with your API key)
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))  # Ensure GOOGLE_API_KEY is set in environment

app = FastAPI()

# Model Selection
MODEL_NAME = "gemini-2.0-flash" #Consider Gemini-1.0-pro for better reasoning, Flash is good for speed but might lack deep analysis.

# Agent Definitions (Simplified for demonstration)
class Agent:
    def __init__(self, name, role, model=MODEL_NAME):
        self.name = name
        self.role = role
        self.model = genai.GenerativeModel(model)

    async def generate_response(self, prompt, history=None):
        try:
            if history:
              response = self.model.generate_content(prompt, generation_config=genai.GenerationConfig(temperature=0.7), safety_settings=[genai.types.SafetySetting(category=genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=genai.types.HarmBlockThreshold.BLOCK_NONE), genai.types.SafetySetting(category=genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=genai.types.HarmBlockThreshold.BLOCK_NONE), genai.types.SafetySetting(category=genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=genai.types.HarmBlockThreshold.BLOCK_NONE), genai.types.SafetySetting(category=genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=genai.types.HarmBlockThreshold.BLOCK_NONE)])
            else:
               response = self.model.generate_content(prompt, generation_config=genai.GenerationConfig(temperature=0.7), safety_settings=[genai.types.SafetySetting(category=genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=genai.types.HarmBlockThreshold.BLOCK_NONE), genai.types.SafetySetting(category=genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=genai.types.HarmBlockThreshold.BLOCK_NONE), genai.types.SafetySetting(category=genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=genai.types.HarmBlockThreshold.BLOCK_NONE), genai.types.SafetySetting(category=genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=genai.types.HarmBlockThreshold.BLOCK_NONE)])

            return response.text
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# Stock/Crypto Analysis Agents
technical_analyst = Agent(name="Technical Analyst", role="Analyzes price charts and technical indicators.")
fundamental_analyst = Agent(name="Fundamental Analyst", role="Analyzes company financials and market trends.")
news_analyst = Agent(name="News Analyst", role="Gathers and analyzes relevant news and sentiment.")
recommendation_agent = Agent(name="Recommendation Agent", role="Consolidates analysis and provides investment recommendations.")

class AnalysisRequest(BaseModel):
    symbol: str
    asset_type: str  # "stock" or "crypto"

class AnalysisResponse(BaseModel):
    technical_analysis: str
    fundamental_analysis: str
    news_analysis: str
    recommendation: str

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_asset(request: AnalysisRequest):
    symbol = request.symbol
    asset_type = request.asset_type

    # Agent interactions (simplified)
    technical_prompt = f"Analyze the technical indicators for {symbol} ({asset_type}). Provide key support and resistance levels, trend analysis, and potential entry/exit points."
    fundamental_prompt = f"Analyze the fundamental factors for {symbol} ({asset_type}). Include financial performance, market position, and future outlook."
    news_prompt = f"Gather and analyze recent news and sentiment surrounding {symbol} ({asset_type}). Identify key drivers and potential risks."
    recommendation_prompt = f"Based on the provided technical, fundamental, and news analysis, provide a comprehensive investment recommendation for {symbol} ({asset_type})."

    tasks = [
        technical_analyst.generate_response(technical_prompt),
        fundamental_analyst.generate_response(fundamental_prompt),
        news_analyst.generate_response(news_prompt),
    ]

    results = await asyncio.gather(*tasks)

    technical_analysis_result = results[0]
    fundamental_analysis_result = results[1]
    news_analysis_result = results[2]

    recommendation_prompt_with_context = recommendation_prompt + "\n\nTechnical Analysis:\n" + technical_analysis_result + "\n\nFundamental Analysis:\n" + fundamental_analysis_result + "\n\nNews Analysis:\n" + news_analysis_result
    recommendation_result = await recommendation_agent.generate_response(recommendation_prompt_with_context)

    return AnalysisResponse(
        technical_analysis=technical_analysis_result,
        fundamental_analysis=fundamental_analysis_result,
        news_analysis=news_analysis_result,
        recommendation=recommendation_result,
    )

# Add this section:
@app.get("/")
async def read_root():
    return {"message": "Welcome to the AssetInsight AI API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)