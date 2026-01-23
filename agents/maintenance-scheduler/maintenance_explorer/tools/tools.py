import datetime
import logging
import os
import re
from ..config import Config
from google.protobuf.json_format import MessageToDict
import altair as alt
import proto
from google.adk.tools import FunctionTool, ToolContext
from google.genai import types
from google.cloud import storage
import io
from google.adk.tools import FunctionTool, ToolContext
from google.cloud import geminidataanalytics
from google.genai.types import Part, Blob
import altair as alt
from io import BytesIO
from typing import Dict, Any
from google.cloud import geminidataanalytics

logger = logging.getLogger(__name__)
data_chat_client = geminidataanalytics.DataChatServiceClient()
configs = Config()



# Define the ADK Function Tool (Must be async since client.chat streams)
async def analytics_chart_tool(
    question: str,
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Queries the Conversational Analytics API, extracts chart data (Vega-Lite), 
    renders it as a PNG, and saves the image to ADK artifact storage.
    """
    try:
        # --- 1. Your Existing API Call Logic ---
        input_message = [geminidataanalytics.Message(
            user_message=geminidataanalytics.UserMessage(text=question)
        )]
        
        # Ensure 'agent_parent' is available in context (e.g., from agent setup)
        parent_name = tool_context.state["agent_parent"]
        data_agent_id = tool_context.state["agent_name"]
        
        billing_project =configs.CLOUD_PROJECT
        location = configs.CLOUD_LOCATION
        conversation_id= tool_context.state["conversation_name"]

        # Create a conversation_reference
        conversation_reference = geminidataanalytics.ConversationReference()
        conversation_reference.conversation = conversation_id
        conversation_reference.data_agent_context.data_agent = data_agent_id

        client = geminidataanalytics.DataChatServiceClient()
        request = geminidataanalytics.ChatRequest(
            messages=input_message,
            parent=parent_name,
            conversation_reference=conversation_reference, # messages field expects a list
        )
        # Use a synchronous client.chat call within an async function
        stream = client.chat(request=request)

        chart_data = None
        # --- 2. Process the Streaming Response ---
        for reply in stream:
            if reply.system_message and reply.system_message.chart:
                if "result" in reply.system_message.chart:
                    # --- 3. Extract and Render the Chart ---
                    # Function to safely convert protobuf map/composite types to Python dicts
                    def _convert_proto(v):
                        if isinstance(v, proto.marshal.collections.maps.MapComposite):
                            return {k: _convert_proto(v) for k, v in v.items()}
                        elif isinstance(v, proto.marshal.collections.RepeatedComposite):
                            return [_convert_proto(el) for el in v]
                        elif isinstance(v, (int, float, str, bool)):
                            return v
                        else:
                            return MessageToDict(v)

                    # Extract the Vega-Lite specification
                    vega_config = _convert_proto(reply.system_message.chart.result.vega_config)
                    
                    # Use Altair to create the chart object from the Vega-Lite spec
                    chart = alt.Chart.from_dict(vega_config)

                    # Save the chart as a PNG to an in-memory buffer
                    buffer = BytesIO()
                    chart.save(buffer, format='png')
                    buffer.seek(0)
                    chart.display()

                    # --- 4. Save the PNG to ADK Artifacts ---
                    # Create the ADK Part object for binary data
                    chart_artifact = Part(
                        inline_data=Blob(
                            mime_type="image/png",
                            data=buffer.read()
                        )
                    )
                    filename = f"analytics_chart.png"                    
                    # Save the artifact using the ToolContext (must use await)
                    version = await tool_context.save_artifact(
                        filename=filename,
                        artifact=chart_artifact
                    )
                    # --- 5. Return the Result ---
                    return {
                        "status": "success",
                        "message": "Chart successfully generated and saved to session artifacts.",
                        "filename": filename,
                        "version": version
                    }

        
        return {"status": "warning", "message": "Query successful, but no chart data was returned by the API."}

    except Exception as e:
        logger.error(f"An error occurred during chart processing: {e}")
        return {"status": "exception", "error": f"An error occurred during chart processing: {e}"}

# # Final ADK Function Tool definition
# analytics_chart_tool = FunctionTool(func=query_and_save_chart)


async def  ask_lakehouse(
    question: str,
    tool_context: ToolContext,
) -> str:

    
    messages = [geminidataanalytics.Message()]
    messages[0].user_message.text = question

    agent_name = tool_context.state["agent_name"] 
    conversation_name = tool_context.state["conversation_name"]
    parent = tool_context.state["agent_parent"]

    billing_project =configs.CLOUD_PROJECT
    # Create a conversation_reference
    conversation_reference = geminidataanalytics.ConversationReference()
    conversation_reference.conversation = conversation_name
    conversation_reference.data_agent_context.data_agent = agent_name

    # Form the request
    request = geminidataanalytics.ChatRequest(
        parent = parent,
        messages = messages,
        conversation_reference = conversation_reference
    )

    # Make the request
    stream = data_chat_client.chat(request=request)
    # Handle the response
    responses = []
    for response in stream:
        responses.append(str(response.system_message))
    
    return responses


# Initialize GCS client (Authenticates using Application Default Credentials)
gcs_client = storage.Client()

async def get_image_from_bucket(
    gs_uri: str, 
    tool_context: ToolContext ) -> dict:
    """
    1. Reads image data from GCS.
    2. Saves the image data as an ADK artifact.
    """
    try:
        # 1. Download the image blob from GCS
        logger.info(f"Attempting to download {gs_uri} from GCS bucket...")
        if not gs_uri.startswith("gs://"):
                raise ValueError("Invalid GCS URI format. Must start with 'gs://'")
            # Remove the "gs://" prefix
        path_without_prefix = gs_uri[len("gs://"):]
                # Split the path into bucket and object parts
        parts = path_without_prefix.split("/", 1)
        if len(parts) < 2:
                raise ValueError("Invalid GCS URI format. Must include bucket and object.")
        bucket_name = parts[0]
        image_name = parts[1]
        file_name= os.path.basename(gs_uri)
        gcs_bucket = gcs_client.bucket(bucket_name)
        gcs_blob = gcs_bucket.blob(image_name)
        if not gcs_blob.exists():
            return {
                "status": "error",
                "message": f"Error: GCS object gs://{bucket_name}/{image_name} not found."
            }

        # Download the content bytes
        image_bytes = gcs_blob.download_as_bytes()
        logger.info(f"Successfully downloaded {len(image_bytes)} bytes.")

        # 2. Create a types.Part object for the artifact
        image_artifact_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/jpeg"
        )

        # 3. Save the Part to the ADK Artifact Service
        # The artifact_service is configured on the ADK Runner.
        version = await tool_context.save_artifact(
            filename=file_name,
            artifact=image_artifact_part
        )
        return f"Artifact saved successfully with filename {file_name}"    

        # 4. Construct a response that the agent can use to show the image
        return {
            "status": "success",
            "message": f"Image '{file_name}' saved to Artifact Service (Version {version}).",
            "image_filename": file_name # This name will be displayed in the UI
        }

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return {
            "status": "error",
            "message": f"An unexpected error occurred: {e}"
        }

# get_image_from_bucket = FunctionTool(
#     save_image_from_gcs,
#     #description="Loads a property image from a Google Cloud Storage bucket and saves it as an artifact for display."
# )


def get_external_url_image( gs_uri: str) -> str:

        """
        Retrieves the public URL of an image from a Google Cloud Storage bucket.

        Args:
            gs_uri (str): uri to the image in a bucket
        Returns:
            str: The public URL of the image, or an error message if not found or accessible.
        """
        try: 
            if not gs_uri.startswith("gs://"):
                raise ValueError("Invalid GCS URI format. Must start with 'gs://'")
            # Remove the "gs://" prefix
            path_without_prefix = gs_uri[len("gs://"):]
            # Split the path into bucket and object parts
            parts = path_without_prefix.split("/", 1)
            if len(parts) < 2:
                raise ValueError("Invalid GCS URI format. Must include bucket and object.")
            bucket_name = parts[0]
            object_name = parts[1]

            # Construct the HTTPS URL
            https_url = f"https://storage.cloud.google.com/{bucket_name}/{object_name}?authuser=1"
            return https_url
        except Exception as e:
            return f"An error occurred while retrieving the image from GCS: {e}"

import matplotlib.pyplot as plt
from datetime import datetime
import io

# # Placeholder for your ADK's artifact saving function
# # In a real ADK environment, this function would handle
# # saving the file and returning a displayable path/URL.
# # For this example, we'll just save it locally.
# def save_artifact(file_content, filename="passenger_plot.png"):
#     """
#     Placeholder for saving the plot to the agent's artifact storage.

#     In a real ADK, this might upload the data and return a URL.
#     Here, it just saves to the local filesystem.
#     """
#     try:
#         with open(filename, 'wb') as f:
#             f.write(file_content)
#         print(f"Plot saved successfully to {filename}")
#         # Return the filename or a URL/path for display
#         return filename
#     except Exception as e:
#         print(f"Error saving artifact: {e}")
#         return None


# async def plot_passenger_data(data_string: str,tool_context: ToolContext) -> dict:
#     """
#     Parses passenger data, plots the number of passengers over time,
#     and saves the plot as an artifact.

#     Args:
#         data_string: A string containing time and passenger data,
#                      e.g., "09/29/2025 21:49: 3 passengers\n..."

#     Returns:
#         A string message containing the path/filename of the saved plot.
#     """
#     try:
#         lines = data_string.strip().split('\n')
#         times = []
#         passengers = []

#         # 1. Parse the Data
#         for line in lines:
#             if not line.strip():
#                 continue
#             try:
#                 # Split at the colon to separate time/date from passenger info
#                 time_str, passenger_info = line.split(': ', 1)
                
#                 # Parse the time string
#                 dt_obj = datetime.strptime(time_str.strip(), '%m/%d/%Y %H:%M')
#                 times.append(dt_obj)

#                 # Extract the number of passengers (assuming format 'X passengers')
#                 count_str = passenger_info.strip().split(' ')[0]
#                 passengers.append(int(count_str))
#             except ValueError as e:
#                 # Handle lines that don't match the expected format
#                 print(f"Skipping malformed line: '{line}'. Error: {e}")
#                 continue

#         if not times:
#             return "Error: Could not parse any valid data points from the input."

#         # 2. Create the Plot
#         plt.figure(figsize=(10, 6))
#         plt.plot(times, passengers, marker='o', linestyle='-', color='indigo')

#         # Add labels and title
#         plt.title('Passenger Count Over Time', fontsize=16)
#         plt.xlabel('Time', fontsize=12)
#         plt.ylabel('Number of Passengers', fontsize=12)
        
#         # Format x-axis to show time clearly
#         plt.gcf().autofmt_xdate() # Auto-formats date labels for better readability
#         plt.grid(True, linestyle='--', alpha=0.7)
#         plt.tight_layout() # Adjust plot to fit all labels

#         # 3. Save the Plot to a buffer and then to artifact storage
#         # Use a BytesIO object to save the figure without writing to a temp file first
#         buf = io.BytesIO()
#         plt.savefig(buf, format='png')
#         plt.close() # Close the figure to free memory

#         # Save the buffer content as an artifact
#         #artifact_path = save_artifact(buf.getvalue(), "passenger_count_plot.png")
#         file_name= "passenger_count_plot.png"
#         version = await tool_context.save_artifact(
#             filename= file_name,
#             artifact= plt
#         )
#                 # 4. Construct a response that the agent can use to show the image
#         return {
#             "status": "success",
#             "message": f"Image '{file_name}' saved to Artifact Service (Version {version}).",
#             "image_filename": file_name # This name will be displayed in the UI
#         }
       
#     except Exception as e:
#         return f"An unexpected error occurred during plotting: {e}"

