p = 'create_climate_notebook.py'
with open(p, 'rb') as handle:
    data = handle.read()
data = data.replace(b'""\r\n)', b'"""\r\n)')
with open(p, 'wb') as handle:
    handle.write(data)
