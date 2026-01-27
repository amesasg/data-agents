# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import sys
import uuid
import vertexai
from vertexai import agent_engines

# Import your agent code
from maintenance_explorer.agent import root_agent
from maintenance_explorer.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Verify versions
import google.cloud.aiplatform
print(f"VERSION: {google.cloud.aiplatform.__version__}")
print(f"PATH:    {vertexai.__file__}")

configs = Config()
STAGING_BUCKET = f"gs://{configs.CLOUD_PROJECT}-maintenance-scheduler-agent-staging"

class AgentWrapper:
    def __init__(self):
        # ⚠️ MOVE AGENT CREATION HERE ⚠️
        # Re-import inside to ensure fresh environment in the cloud
        from agent import root_agent  # Change 'agent' to your actual file name
        
        print("Initializing agent inside cloud container...")
        self.agent = root_agent 

    def query(self, message: str, **kwargs):
        """Entry point for Vertex AI."""
        # Filter kwargs to only what your agent supports
        call_kwargs = {}
        if "session_id" in kwargs:
            call_kwargs["session_id"] = kwargs["session_id"]
        if "user_id" in kwargs:
            call_kwargs["user_id"] = kwargs["user_id"]
            
        return self.agent(message=message, **call_kwargs)

# Initialize Vertex AI SDK
vertexai.init(
    project=configs.CLOUD_PROJECT,
    location=configs.CLOUD_LOCATION,
    staging_bucket=STAGING_BUCKET,
)

parser = argparse.ArgumentParser(description="Bus stop maintenance scheduler app")
parser.add_argument("--delete", action="store_true", help="Delete deployed agent")
parser.add_argument(
    "--resource_id",
    required="--delete" in sys.argv,
    help="The resource id of the agent to be deleted (projects/.../locations/.../reasoningEngines/...)",
)

args = parser.parse_args()

if args.delete:
    try:
        logging.info(f"Attempting to delete: {args.resource_id}")
        # --- FIX: Use the class constructor to get a handle to the remote resource ---
        remote_app = agent_engines.ReasoningEngine(args.resource_id)
        remote_app.delete()
        logging.info(f"Agent {args.resource_id} deleted successfully")
    except NotFound:
        logging.error(f"Agent {args.resource_id} not found")
    except Exception as e:
        logging.error(f"Error deleting agent: {e}")

else:
    logging.info("Deploying agent to Reasoning Engine...")
    app = agent_engines.AdkApp(agent=root_agent)
    
    # --- FIX: Create using the Wrapper Class ---
    # We pass an INSTANCE of the wrapper, not the class itself.
    remote_app = agent_engines.create(
        app, 
        requirements="./maintenance_explorer/requirements.txt",
        display_name="Bus Maintenance Scheduler v1.1",
        description="Agent to assist with bus maintenance scheduling",
        # Ensure your local files are uploaded so 'maintenance_explorer' is available remotely
        extra_packages=["./"], 
    )

    logging.info(f"Agent deployed successfully: {remote_app.resource_name}")

    # Test the deployment
    user_id = "user"
    session_id = f"test-session-{uuid.uuid4()}"

    logging.debug("Testing deployment query...")
    try:
        response = remote_app.query(
            message="Is now a weekend?",
            user_id=user_id,
            session_id=session_id
        )
        logging.info(f"Test Response: {response}")
    except Exception as e:
        logging.error(f"Test query failed: {e}")