import streamlit as st
import time
import numpy as np
import pandas as pd

st.title('Stock Trading Application')
st.divider()
st.header('Introduction')
st.text('Here is some text about the code')
st.divider()
st.header('Next Part')
st.text('Here is some text about the next part')
st.divider()
st.header("Test Section")


_CODE_SNIPPET = """
def greet(name):
    print(f"Hello, {name}!")

greet("Streamlit User")
"""

if st.button("Live stream code"):
    placeholder = st.empty()
    code_accum = ""
    for char in _CODE_SNIPPET:
        code_accum += char
        placeholder.code(code_accum, language="python")
        time.sleep(0.02)

