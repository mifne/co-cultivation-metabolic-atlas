import urllib.request,tarfile,io,pathlib,json
p=pathlib.Path('/home/reiya/co-cultivation/outputs/metabolic_map_20260922/vendor')
t=tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen('https://registry.npmjs.org/escher/-/escher-1.8.2.tgz').read()))
for src,dst in [('package/dist/escher.min.js','escher.min.js'),('package/LICENSE','ESCHER_LICENSE')]:
 (p/dst).write_bytes(t.extractfile(src).read())
print('Escher 1.8.2 vendored')
