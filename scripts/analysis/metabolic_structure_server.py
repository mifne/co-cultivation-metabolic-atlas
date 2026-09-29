"""Local structure resolver: exact annotated InChIKey -> PubChem InChI -> RDKit SVG."""
import json,urllib.request,urllib.parse,time,threading,hashlib
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/metabolic_map_20260922'; CACHE=OUT/'structures';CACHE.mkdir(exist_ok=True)
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
class Handler(BaseHTTPRequestHandler):
 def do_GET(self):
  q=urllib.parse.urlparse(self.path);p=urllib.parse.parse_qs(q.query)
  if q.path=='/health':result={'status':'ok'}
  elif q.path=='/ranking':
   from metabolic_fba_service import check
   try:result=check(p.get('species',[''])[0],'',1,ranking=True,medium_override=json.loads(p['medium'][0]) if 'medium' in p else None)
   except (ValueError,TypeError):result={'status':'unknown','message':'培地条件の形式が不正です'}
  elif q.path=='/fba':
   from metabolic_fba_service import check
   try:result=check(p.get('species',[''])[0],p.get('reaction',[''])[0],int(p.get('sign',['1'])[0]),medium_override=json.loads(p['medium'][0]) if 'medium' in p else None)
   except Exception:result={'status':'unknown','message':'計算サービス内でエラーが発生しました'}
  elif q.path=='/structure':result=structure(p.get('species',[''])[0],p.get('met',[''])[0])
  else:self.send_error(404);return
  self.send_response(200);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Access-Control-Allow-Origin','http://localhost:8768');self.end_headers();self.wfile.write(json.dumps(result,ensure_ascii=False).encode())
if __name__=='__main__':ThreadingHTTPServer(('127.0.0.1',8769),Handler).serve_forever()
