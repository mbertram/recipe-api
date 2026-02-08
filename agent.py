import asyncio
import logging
import os
import sys

from dotenv import load_dotenv
from github import Github, Auth
from llama_index.core.agent.workflow import (
    FunctionAgent,
    AgentWorkflow,
    AgentOutput,
    ToolCall,
    ToolCallResult,
)
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI

load_dotenv()

if len(sys.argv) > 1:
    os.environ["GITHUB_TOKEN"] = sys.argv[1]
    os.environ["REPOSITORY"] = sys.argv[2]
    os.environ["PR_NUMBER"] = sys.argv[3]
    os.environ["OPENAI_API_KEY"] = sys.argv[4]
    if len(sys.argv) > 5:
        os.environ["OPENAI_BASE_URL"] = sys.argv[5]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

repo_url = "https://github.com/mbertram/recipe-api.git"

github_token = os.getenv("GITHUB_TOKEN")
repository = os.getenv("REPOSITORY")
pr_number = os.getenv("PR_NUMBER")

auth = Auth.Token(github_token)
git = Github(auth=auth)

if repository:
    full_repo_name = repository
else:
    repo_name = repo_url.split('/')[-1].replace('.git', '')
    username = repo_url.split('/')[-2]
    full_repo_name = f"{username}/{repo_name}"

logger.info("Connecting to repository: %s", full_repo_name)
repo = git.get_repo(full_repo_name)
logger.info("Successfully connected to repository")

llm = OpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
)
logger.info("LLM initialized with model: %s", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))


def get_pr_details(pr_number: int) -> dict:
    """Fetch pull request details given the PR number. Returns author, title, body, diff_url, state, and commit SHAs."""
    logger.info("Fetching PR details for PR #%d", pr_number)
    pr = repo.get_pull(pr_number)
    commit_shas = [c.sha for c in pr.get_commits()]
    result = {
        "user": pr.user.login,
        "title": pr.title,
        "body": pr.body,
        "diff_url": pr.diff_url,
        "state": pr.state,
        "commit_shas": commit_shas,
        "head_sha": commit_shas[-1] if commit_shas else None,
    }
    logger.info("PR #%d: title=%s, state=%s, commits=%d", pr_number, pr.title, pr.state, len(commit_shas))
    return result


def get_pr_commit_details(commit_sha: str) -> list:
    """Fetch commit details given a commit SHA. Returns a list of changed files with filename, status, additions, deletions, changes, and patch."""
    logger.info("Fetching commit details for SHA: %s", commit_sha[:8])
    commit = repo.get_commit(commit_sha)
    changed_files = []
    for f in commit.files:
        changed_files.append({
            "filename": f.filename,
            "status": f.status,
            "additions": f.additions,
            "deletions": f.deletions,
            "changes": f.changes,
            "patch": f.patch,
        })
    logger.info("Commit %s: %d files changed", commit_sha[:8], len(changed_files))
    return changed_files


def get_file_content(file_path: str) -> str:
    """Fetch the contents of a file from the repository given its file path."""
    logger.info("Fetching file content: %s", file_path)
    content = repo.get_contents(file_path)
    decoded = content.decoded_content.decode("utf-8")
    logger.info("File %s: %d bytes", file_path, len(decoded))
    return decoded


def post_review_to_github(pr_number: int, comment: str) -> str:
    """Post a review comment to a GitHub pull request given the PR number and the review comment body."""
    logger.info("Posting review to PR #%d", pr_number)
    pr = repo.get_pull(pr_number)
    pr.create_review(body=comment)
    logger.info("Review posted successfully to PR #%d", pr_number)
    return f"Review posted successfully to PR #{pr_number}."


async def add_context_to_state(ctx, context: str) -> str:
    """Useful for adding the gathered context to the state."""
    logger.info("Adding context to state (%d chars)", len(context))
    ctx.state["gathered_contexts"] = context
    return "State updated with gathered contexts."


async def add_comment_to_state(ctx, draft_comment: str) -> str:
    """Useful for adding the draft comment to the state."""
    logger.info("Adding draft comment to state (%d chars)", len(draft_comment))
    ctx.state["review_comment"] = draft_comment
    return "State updated with draft comment."


async def add_final_review_to_state(ctx, final_review: str) -> str:
    """Useful for adding the final review comment to the state."""
    logger.info("Adding final review to state (%d chars)", len(final_review))
    ctx.state["final_review_comment"] = final_review
    return "State updated with final review comment."


pr_details_tool = FunctionTool.from_defaults(get_pr_details)
pr_commit_details_tool = FunctionTool.from_defaults(get_pr_commit_details)
file_content_tool = FunctionTool.from_defaults(get_file_content)
post_review_tool = FunctionTool.from_defaults(post_review_to_github)

CONTEXT_SYSTEM_PROMPT = """You are the context gathering agent. When gathering context, you MUST gather:
    - The details: author, title, body, diff_url, state, and head_sha;
    - Changed files;
    - Any requested for files;
Once you gather the requested info, you MUST hand control back to the Commentor Agent."""

context_agent = FunctionAgent(
    llm=llm,
    name="ContextAgent",
    description="Gathers all the needed context from the GitHub repository including PR details, changed files, and file contents.",
    tools=[pr_details_tool, pr_commit_details_tool, file_content_tool, add_context_to_state],
    system_prompt=CONTEXT_SYSTEM_PROMPT,
    can_handoff_to=["CommentorAgent"],
)

COMMENTOR_SYSTEM_PROMPT = """You are the commentor agent that writes review comments for pull requests as a human reviewer would.
Ensure to do the following for a thorough review:
 - Request for the PR details, changed files, and any other repo files you may need from the ContextAgent.
 - Once you have asked for all the needed information, write a good ~200-300 word review in markdown format detailing:
    - What is good about the PR?
    - Did the author follow ALL contribution rules? What is missing?
    - Are there tests for new functionality? If there are new models, are there migrations for them? - use the diff to determine this.
    - Are new endpoints documented? - use the diff to determine this.
    - Which lines could be improved upon? Quote these lines and offer suggestions the author could implement.
 - If you need any additional details, you must hand off to the Context Agent.
 - You should directly address the author. So your comments should sound like:
 "Thanks for fixing this. I think all places where we call quote should be fixed. Can you roll this fix out everywhere?"
 - You must hand off to the ReviewAndPostingAgent once you are done drafting a review.
"""

commentor_agent = FunctionAgent(
    llm=llm,
    name="CommentorAgent",
    description="Uses the context gathered by the context agent to draft a pull review comment.",
    tools=[add_comment_to_state],
    system_prompt=COMMENTOR_SYSTEM_PROMPT,
    can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"],
)

REVIEW_AND_POSTING_SYSTEM_PROMPT = """You are the Review and Posting agent. You must use the CommentorAgent to create a review comment.
Once a review is generated, you need to run a final check and post it to GitHub.
   - The review must:
   - Be a ~200-300 word review in markdown format.
   - Specify what is good about the PR:
   - Did the author follow ALL contribution rules? What is missing?
   - Are there notes on test availability for new functionality? If there are new models, are there migrations for them?
   - Are there notes on whether new endpoints were documented?
   - Are there suggestions on which lines could be improved upon? Are these lines quoted?
 If the review does not meet this criteria, you must ask the CommentorAgent to rewrite and address these concerns.
 When you are satisfied, post the review to GitHub."""

review_and_posting_agent = FunctionAgent(
    llm=llm,
    name="ReviewAndPostingAgent",
    description="Reviews the draft comment generated by the CommentorAgent, checks quality, and posts the final review to GitHub.",
    tools=[add_final_review_to_state, post_review_tool],
    system_prompt=REVIEW_AND_POSTING_SYSTEM_PROMPT,
    can_handoff_to=["CommentorAgent"],
)

workflow_agent = AgentWorkflow(
    agents=[context_agent, commentor_agent, review_and_posting_agent],
    root_agent=review_and_posting_agent.name,
    initial_state={
        "gathered_contexts": "",
        "review_comment": "",
        "final_review_comment": "",
    },
)


async def main():
    query = f"Write a review for PR number {pr_number}"
    logger.info("Query: %s", query)
    prompt = RichPromptTemplate(query)

    handler = workflow_agent.run(prompt.format())

    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            logger.info("Agent switch: %s", current_agent)
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                logger.info("Agent output received (%d chars)", len(event.response.content))
                print("\n\nFinal response:", event.response.content)
            if event.tool_calls:
                print("Selected tools: ", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCallResult):
            logger.debug("Tool output: %s", event.tool_output)
            print(f"Output from tool: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")

    logger.info("Query processing complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        git.close()
        logger.info("GitHub connection closed")