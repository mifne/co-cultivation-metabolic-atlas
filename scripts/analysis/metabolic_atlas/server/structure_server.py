"""Local structure resolver: exact annotated InChIKey -> PubChem InChI -> RDKit SVG."""
import json,urllib.request,urllib.parse,time,threading,hashlib,argparse,mimetypes
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
ROOT=Path(__file__).resolve().parents[4]; OUT=ROOT/'outputs/metabolic_map_20260922'; CACHE=OUT/'structures';CACHE.mkdir(exist_ok=True)
D=json.loads((OUT/'model_data.json').read_text()); SPECIES={s['short']:s for s in D['species']};gate=threading.Lock()
def structure(short,mid):
 m=SPECIES.get(short,{}).get('metabolites',{}).get(mid)
 if not m:return {'status':'missing','message':'モデル内に未登録'}
 key=m.get('annotation',{}).get('inchikey');key=key[0] if isinstance(key,list) and len(key)==1 else key
 if not isinstance(key,str):return {'status':'missing','message':'一意なInChIKeyが未登録です'}
 file=CACHE/(hashlib.sha256(key.encode()).hexdigest()+'.json')
 if file.exists():return json.loads(file.read_text())
 url='https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/'+urllib.parse.quote(key,safe='')+'/property/InChI/JSON'
 try:
  with gate:
   req=urllib.request.Request(url,headers={'User-Agent':'LocalMetabolicAtlas/1.0'})
   with urllib.request.urlopen(req,timeout=12) as res:data=json.load(res)
   time.sleep(.3)
  matches=[]
  for p in data['PropertyTable']['Properties']:
   mol=Chem.MolFromInchi(p['InChI'])
   if mol is not None and Chem.MolToInchiKey(mol)==key:matches.append((mol,p))
  if not matches:return {'status':'unmatched','message':'構造が注釈InChIKeyに一致しません'}
  mol,p=matches[0];draw=rdMolDraw2D.MolDraw2DSVG(320,220);draw.DrawMolecule(mol);draw.FinishDrawing()
  result={'status':'ok','svg':draw.GetDrawingText(),'inchikey':key,'cid':p['CID'],'source':'https://pubchem.ncbi.nlm.nih.gov/compound/'+str(p['CID']),'generator':'RDKit '+__import__('rdkit').__version__}
  file.write_text(json.dumps(result,ensure_ascii=False));return result
 except Exception as exc:return {'status':'unavailable','message':'構造の取得に失敗しました。接続状態や注釈を確認してください。','error_type':type(exc).__name__}
from atlas_store import JobQueue, SessionStore
from fba_service import check, read_model_data

def calculate_job(body):
 return check(body['species'],body.get('reaction',''),body.get('sign',1),ranking=body['kind']=='ranking',fva=body['kind']=='fva',medium_override=body.get('medium'))
jobs=JobQueue(calculate_job)
sessions=SessionStore(OUT/'sessions',lambda:read_model_data()[1])

class Handler(BaseHTTPRequestHandler):
 def json_response(self,result,status=200):
  raw=json.dumps(result,ensure_ascii=False,allow_nan=False).encode()
  self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(raw)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(raw)
 def do_POST(self):
  try:
   # Same-origin writes; keep local endpoints out of cross-site form submissions.
   origin=self.headers.get('Origin')
   if origin and urllib.parse.urlparse(origin).netloc!=self.headers.get('Host'):self.json_response({'message':'Origin mismatch'},403);return
   if self.headers.get('Content-Type','').split(';')[0]!='application/json':self.json_response({'message':'JSON required'},415);return
   size=int(self.headers.get('Content-Length','0'))
   if size<=0 or size>2_000_000:self.json_response({'message':'Invalid request size'},413);return
   body=json.loads(self.rfile.read(size))
   path=urllib.parse.urlparse(self.path).path
   if path=='/api/jobs':
    if not isinstance(body,dict) or body.get('kind') not in ('ranking','fba','fva') or body.get('species') not in SPECIES:raise ValueError('計算条件が不正です')
    if body.get('sign',1) not in (-1,1) or isinstance(body.get('sign'),bool):raise ValueError('反応方向が不正です')
    result=jobs.submit(body)
   elif path.startswith('/api/jobs/') and path.endswith('/cancel'):result=jobs.cancel(path.split('/')[-2])
   elif path=='/api/sessions':result=sessions.save(body)
   else:self.send_error(404);return
   self.json_response(result)
  except (ValueError,TypeError) as exc:self.json_response({'message':str(exc)},400)
  except KeyError:self.json_response({'message':'計算が見つかりません'},404)
 def do_GET(self):
  path=urllib.parse.urlparse(self.path).path
  try:
   if path=='/api/model':self.json_response({'model':read_model_data()[1]});return
   if path=='/api/sessions':self.json_response({'sessions':sessions.list()});return
   if path.startswith('/api/sessions/'):self.json_response(sessions.load(path.rsplit('/',1)[1]));return
   if path.startswith('/api/jobs/'):self.json_response(jobs.get(path.rsplit('/',1)[1]));return
  except (ValueError,TypeError) as exc:self.json_response({'message':str(exc)},400);return
  except (KeyError,FileNotFoundError):self.json_response({'message':'保存状態または計算が見つかりません'},404);return
  self.legacy_get()
 def legacy_get(self):
  q=urllib.parse.urlparse(self.path);p=urllib.parse.parse_qs(q.query)
  if q.path in ('/','/index.html') or q.path in ('/vendor/cytoscape.min.js','/vendor/escher.min.js'):
   file=OUT/('index.html' if q.path in ('/','/index.html') else q.path.lstrip('/'))
   if not file.is_file():self.send_error(404);return
   raw=file.read_bytes();self.send_response(200);self.send_header('Content-Type',mimetypes.guess_type(file.name)[0] or 'application/octet-stream');self.send_header('Content-Length',str(len(raw)));self.send_header('Cache-Control','no-cache');self.end_headers();self.wfile.write(raw);return
  if q.path.startswith('/api/'):q=q._replace(path=q.path[4:])
  if q.path=='/health':result={'status':'ok'}
  elif q.path=='/ranking':
   from fba_service import check
   try:result=check(p.get('species',[''])[0],'',1,ranking=True,medium_override=json.loads(p['medium'][0]) if 'medium' in p else None)
   except (ValueError,TypeError):result={'status':'unknown','message':'培地条件の形式が不正です'}
  elif q.path=='/fba':
   from fba_service import check
   try:result=check(p.get('species',[''])[0],p.get('reaction',[''])[0],int(p.get('sign',['1'])[0]),medium_override=json.loads(p['medium'][0]) if 'medium' in p else None)
   except Exception:result={'status':'unknown','message':'計算サービス内でエラーが発生しました'}
  elif q.path=='/structure':result=structure(p.get('species',[''])[0],p.get('met',[''])[0])
  else:self.send_error(404);return
  self.send_response(200);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Access-Control-Allow-Origin','http://localhost:8768');self.end_headers();self.wfile.write(json.dumps(result,ensure_ascii=False).encode())
if __name__=='__main__':
 parser=argparse.ArgumentParser(description='Metabolic atlas and same-origin API')
 parser.add_argument('--bind',default='127.0.0.1');parser.add_argument('--port',type=int,default=8768)
 args=parser.parse_args();ThreadingHTTPServer((args.bind,args.port),Handler).serve_forever()
