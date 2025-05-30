from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

my_credentials = {
    "url" : "https://us-south.ml.cloud.ibm.com"
}

params = {
    GenParams.MAX_NEW_TOKENS: 700, # The maxium number of tokens that the model can generate in a single run.
    GenParams.TEMPERATURE: 0.1, # A parameter that 
}

LLAMA2_model = Model(
    model_id = 'meta-llama/llama-3-2-11b-vision-instruct',
    credentials = my_credentials,
    params = params,
    project_id = "skills-network",
)

llm = WatsonxLLM(LLAMA2_model)

print(llm("How to read a book effectively?"))