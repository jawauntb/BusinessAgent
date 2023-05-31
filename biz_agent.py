from langchain.chains import SimpleSequentialChain
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain import OpenAI, LLMChain
from langchain.chains import LLMRequestsChain
from langchain.prompts import PromptTemplate
from langchain.agents import load_tools
from langchain.chains.conversation.memory import ConversationBufferMemory
from dotenv import load_dotenv
from llama_index import SimpleWebPageReader
import requests
from bs4 import BeautifulSoup
from langchain.agents.conversational.base import ConversationalAgent
# initial commit (this is pretty old but hadn't put on gh yet)
load_dotenv()

def chain_llm_chains(*llm_chains):
    return SimpleSequentialChain(chains=[chain for chain in llm_chains], verbose=True)


def create_chain(llm, template):
    prompt = PromptTemplate(
        input_variables=["input"], template=template)
    return LLMChain(llm=llm, prompt=prompt, verbose=True)


def create_multi_prompt_chain(prompts):
    prompts = []
    llm_chains = []
    for prompt in prompts:
        llm = OpenAI(
            temperature=0.65, max_tokens=2000, presence_penalty=0.5, frequency_penalty=0.5, verbose=True)
        chain = create_chain(llm, prompt)
        llm_chains.append(chain)
    return chain_llm_chains(*llm_chains)

# create prompts

task_head = """
    You are a super smart business model and product expert that is great at
    taking assumptions underlying business models, discovering how risky, impactful and uncertain each
    assumption is, and testing the most critical assumptions using cheap, fast experiments that leverage
    data we collected from our email list, social media channels and in store data.

    You are composing a set of hypotheses, and one by one, we will take each aspect of the business model
    and analyze them.
"""

business_components = """
    The aspects of a business model canvas are: Customer Segment, Channels, Relationships,and Value
    propositions (which are based on Value Maps that map to Customer Profiles). These pertain to the
    business or product's desirability. The feasibility aspects are Key Resources, Key Activities and Key
    Partners. The viability aspect includes Revenue Sources and Profits, and finally, Cost Structures.
"""

task_context_tail = """

    Determine whether the idea in the canvas belongs to the desirability, feasibility, or
    viability category, return each individual idea or statement as an object labeled like this example:
    (
        Domain: Customer Segment,
        Category: Desirability,
        Statement: “Our email list subscribers are the ideal segment for our new product”
    )

    You will be given a Business Model Canvas with 9 categories, and you will need to determine the category for each
    canvas element. {input}:
"""

task_context_prompt = task_head + business_components + task_context_tail

assumption_mapping_hierarchy = """
[
    Business Assumption Priority Hierarchy
    (most important, get this right first) Desirability: Customer Segment, Value Proposition, Channels, Relationships
    Feasibility: Key Resources, Key Activities, Key Partners
    Viability: Revenues, Costs
]
"""

assumption_mapping_segments = """
You are taking the state,en a section of the business model canvas, and you are going to map the assumptions in it based on the criteria for that section.

Desirability, Feasibility, Viability Criteria

Desirability Criteria:
    Customer Segments
        Right Segment?
        Segment(s) Exist?
        Big Enough Segment(s)?

    Customer Relationships
        Can build the right relationships?
        Difficult for customer to switch?
        Can retain customers?

    Value Propositions
        Right value for customer segment?
        Unique and hard to replicate
        Value Map:
            high value jobs, important gains, top pains
        Customer Profile:
            jobs, pains & gains really matter to customer


Feasibility Criteria:
    Key Activities
        Can perform at scale
        Can execute with the right quality

    Key Resources
        Can secure and manage all tech and resources at scale
        includes IP, human & financial capital

    Key Partners
        Can create the right partnerships


Viability Criteria:
    Revenues
        Can get a specific price?
        Can generate sufficient $$?

    Costs
        Can manage costs from infrastructure and keep them under control?

    Profits
        Can generate more revenues than costs?
"""

assumption_mapping_task = """
You are the worlds best business modeling expert that knows how to uncover the assumptions underlying a business idea.

Identify the central assumptions underlying the business idea object: {object}.
mapped to its underlying assumptions as a field in the object.
Only write assumptions for the area it belongs to.

Example Input:
    "Customer Segments
            The core customer segment are people that signed up for our web3 or oasis email lists, followed by:
            Water NFT Holders That came in person to mint
            Twitter Followers
            IG Followers
            Bed-Stuy community locals
    "
In our example, we would only worry about the category (*desirability* in this case) and the subsequent assumptions that matter
ONLY for the *currently selected domain (in this example its customer segment) that need to be right given our customer segment.
For Customer Segments and Desirability example, our assumptions should justify the following criteria:
        Is it the right segment?
        do(es) the segment(s) exist?
        is it a big enough segment(s)?
).

Given these things need to be true, formulate the assumptions that would make them true.
In our example, we might write something like these as the assumptions:
    "The email list subscribers are the right segment to build for",
    "We understand what is valuable to the email list subscribers",
    "We can reach the email list subscribers to share out product",
    "The email list subscribers are willing to pay for our product"

finally, return the updated idea inside its category, (eg:
   Category: Desirability:
    Domain: Customer Segment:
        Statement: "Our email list subscribers are the ideal segment for our new product”
            Assumptions: [
                "The email list subscribers are the right segment to build for",
                "We understand what is valuable to the email list subscribers",
                "We can reach the email list subscribers to share out product",
                "The email list subscribers are willing to pay for our product"
            ]
"""

assumption_mapping_prompt =  assumption_mapping_hierarchy + assumption_mapping_segments + assumption_mapping_task

hypothesizing_prompt = """

EXPERIMENT STRUCTURE:
Hypothesis Test/Experiment Name
Deadline, Duration
We Believe that: [Risk(Uncertainty * Impact)]
To Verify, we will: [Cost] [Data Reliability]
And Measure: [Time Req'd]
We are Right if:

*[] = rating

Generate 5 potential tests and assign them a value from 1-5 to indicate ease, cost, quickness, strength of evidence
"""


def create_task_chain(template):
    llm = OpenAI(
        temperature=0.65, max_tokens=2500, presence_penalty=0.5, frequency_penalty=0)
    prompt = PromptTemplate(
        input_variables=["input"], template=template)
    return LLMChain(llm=llm, prompt=prompt, verbose=True)


def create_assumption_chain(template):
    llm = OpenAI(
        temperature=0.6, max_tokens=2500, presence_penalty=0.725, frequency_penalty=0.25)
    prompt = PromptTemplate(
        input_variables=["object"], template=template)
    return LLMChain(llm=llm, prompt=prompt, verbose=True)


task = create_task_chain(task_context_prompt)
assumption = create_assumption_chain(assumption_mapping_prompt)
chain_list = [task, assumption]

BMC_mapping = chain_llm_chains(*chain_list)

key_partners = """
Key Partners
        Special Guests,
        Special Service Providers
"""

key_activities = """
Key Activities
        Marketing to our customer segments
        Pricing The NFT + Offering
        Updating the Mint Site
        Deploying the NFT and Updated Site
        Developing the NFT (smart contract, artwork)
        Communicating with Holders and Prospects
        Planning + Executing Web3 Experiences for holders
"""

key_resources = """
 Key Resources
        Oasis, Seed Pod, Projection Mapping, etc.
        Event Throwing Infrastructure (PMs, Producers, Creative Dir., IT)
        Product + Growth actor
        Website, NFT Contract, Metadata, Artwork
"""

value_propositions = """
Value Propositions
    Surprise giveaways & contests
    Creative IRL workshops & activations
    POAP (Proof of Attendance Protocol) Rewards
    Holders get to plant a SEED in the future
"""

customer_relationships = """
Customer Relationships
    Build:
        Sharing on twitter, in our discord to current members, encouraging purchase/wallet setup at the oasis,
    directing people to our mint site or to an in app mint on instagram
    Maintain:
        We add them to Discord and message periodically to catch up or share special opportunities with them
    Nature of relationship:
        they come in to the store to access rewards and experiences, staff is very hype to see them, we track
        who claimed or hasn't claimed their rewards,

"""

channels = """
Channels
    Communication:
        Seed XYZ site
        BA in Oasis
        Twitter
        Instagram
        Discord
        TikTok

    Sales:
        Seed XYZ site
        BA in Oasis
        Instagram

    Distribution:
        Seed XYZ site
        BA in Oasis
"""

customer_segments = """
Customer Segments
        The core customer segment are people that signed up for our web3 or oasis email lists, followed by:
        Water NFT Holders That came in person to mint
        Twitter Followers
        IG Followers
        Bed-Stuy community locals
"""

revenue_streams = """
    Revenue Sources
        We believe that we can cover production costs because we already have the digital infrastructure to produce NFTs,
        and we believe that we can optimally target our customer segments in order to market the NFT.
        We also have a bigger following to engage with this time around, which reduces customer acquisition costs some amount
        NFT Sales
        Add-on, following or recurring sales to holders
        activations or other $ary opportunities downstream that result from attention this may generate
"""

cost_structure = """
    Cost Structure
        Fixed Costs:
            NFT Production (economy of scale)
        Variable Costs:
            Customer Acquisition
"""

canvas_domains = [key_partners, key_activities, key_resources, value_propositions, customer_relationships, channels, customer_segments, revenue_streams, cost_structure]

business_model_canvas = """
Feasibility:

    Key Partners
        Special Guests,
        Special Service Providers

    Key Activities
        Marketing to our customer segments
        Pricing The NFT + Offering
        Updating the Mint Site
        Deploying the NFT and Updated Site
        Developing the NFT (smart contract, artwork)
        Communicating with Holders and Prospects
        Planning + Executing Web3 Experiences for holders

    Key Resources
        Oasis, Seed Pod, Projection Mapping, etc.
        Event Throwing Infrastructure (PMs, Producers, Creative Dir., IT)
        Product + Growth actor
        Website, NFT Contract, Metadata, Artwork


Desirability:

Value Propositions
    Surprise giveaways & contests
    Creative IRL workshops & activations
    POAP (Proof of Attendance Protocol) Rewards
    Holders get to plant a SEED in the future.
    (
        The value prop offers:
        Surprise giveaways & contests
        Creative IRL workshops & activations
        POAP (Proof of Attendance Protocol) Rewards
        for our customer segments
        Most importantly to resolve this pain:
        And also to resolve these pains:
        and it also provides this important gain:
        Among other gains including:
    )

Customer Relationships
    Build:
        Sharing on twitter, in our discord to current members, encouraging purchase/wallet setup at the oasis,
    directing people to our mint site or to an in app mint on instagram
    Maintain:
        We add them to Discord and message periodically to catch up or share special opportunities with them
    Nature of relationship:
        they come in to the store to access rewards and experiences, staff is very hype to see them, we track
        who claimed or hasn't claimed their rewards,

Customer Segments
    Bed-Stuy Community
    Water NFT Holders That came in person to mint
    People on web3 email list
    (
        The core customer segment is web3 email list members, followed by:
        Water NFT holders
        Twitter Followers
        IG Followers
        Bed-Stuy locals
    )

Channels
    Communication:
        Seed XYZ site
        BA in Oasis
        Twitter
        Instagram
        Discord
        TikTok

    Sales:
        Seed XYZ site
        BA in Oasis
        Instagram

    Distribution:
        Seed XYZ site
        BA in Oasis


Viability:

    Cost Structure
        Fixed Costs:
            NFT Production (economy of scale)
        Variable Costs:
            Customer Acquisition

    Revenue Sources
        We believe that we can cover production costs because we already have the digital infrastructure to produce NFTs,
        and we believe that we can optimally target our customer segments in order to market the NFT.
        We also have a bigger following to engage with this time around, which reduces customer acquisition costs some amount
        NFT Sales
        Add-on, following or recurring sales to holders
        activations or other $ary opportunities downstream that result from attention this may generate
"""

# create business process mapping chain
filename = "/Users/jawaun/langchain_stuff/playing_around/docs/assumptions/assumption_mapping_"

# with open(file, "w") as f:
#     for domain in canvas_domains:
#         output = BMC_mapping.run(domain)
#         print(output)
#         f.write(output)


solo_task = """
Identify the key assumptions underlying this business idea: {input}.
A good assumption is testable, concise and is necessarily true if the idea is to be successful or executed upon.
finally, return the idea and a map of its category, domain, and assumptions, like so [example]:

ONLY worry about the criteria for assumptions WITHIN the same category and domain. ie segments, channels, value props only focus on desirability criteria, resources, activities, partners only focus on feasibility criteria, and revenue streams, cost structure only focus on viability criteria.

Category: Desirability:
    Domain: Customer Segment:
        Statement: "Our email list subscribers are the ideal segment for our new product”
            Assumptions: [
                "The email list subscribers are the right segment to build for",
                "We understand what is valuable to the email list subscribers",
                "We can reach the email list subscribers to share out product",
                "The email list subscribers are willing to pay for our product"
            ]
"""

solo_task_mod = """
Identify the key assumptions underlying this business idea: {input}.
ONLY worry about the criteria for assumptions WITHIN the same category and domain. ie segments, channels, value props only focus on desirability criteria, resources, activities, partners only focus on feasibility criteria, and revenue streams, cost structure only focus on viability criteria.
A good assumption is testable, concise and is necessarily true if the idea is to be successful or executed upon.
finally, return the idea and a map of its category, domain, and assumptions, like so [example]:
    Statement: "email list subscribers are the ideal segment for our new product”
    Assumptions: [
        "The email list subscribers are the right segment to build for",
        "We understand what is valuable to the email list subscribers",
        "We can reach the email list subscribers to share out product",
        "The email list subscribers are willing to pay for our product"
    ]
"""
solo_assumer = assumption_mapping_hierarchy + assumption_mapping_segments + solo_task
llm = OpenAI(
    temperature=0.6, max_tokens=1000, presence_penalty=0.5, frequency_penalty=0.5)
prompt = PromptTemplate(
    input_variables=["input"], template=solo_assumer)

assumer = LLMChain(llm=llm, prompt=prompt)
# canvas_domains = [, value_propositions,
#                   customer_relationships, channels, customer_segments, revenue_streams, cost_structure]
feasible = [key_partners, key_activities, key_resources]
desirable = [value_propositions, customer_relationships, channels, customer_segments]
viable = [revenue_streams, cost_structure]
# areas = [feasible, desirable, viable]
areas = {
    "desirable": desirable,
    "feasible": feasible,
    "viable": viable
}

for area in areas:
    fname = filename + area + ".txt"
    with open(fname, "w") as fi:
        for domain in areas[area]:
            output = assumer.run(domain)
            fi.write(f"{domain}:  {output}\n")
            print('done with domain', domain)
        fi.close()
    print('finished section', area)

