# Use the official Pydantic v2 converter as-is
from temporalio.contrib.pydantic import pydantic_data_converter

# This returns a temporalio.converter.DataConverter instance
data_converter = pydantic_data_converter
