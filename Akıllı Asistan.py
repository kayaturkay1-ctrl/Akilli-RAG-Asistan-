import os
import faiss
from sentence_transformers import SentenceTransformer
import torch
import pymupdf as fitz
from openai import OpenAI
import pickle
import gradio as gr

os.environ["OPENAI_API_KEY"] = ""#API key inizi buraya yazınız. / Wrıte API key here.

# ==========================================
# 1. PDF OKUMA FONKSİYONU
# ==========================================
def pdf_okuyucu(pdf_yolu, parent_boyutu=1000, parent_kesisim=200, child_boyutu=250, child_kesisim=50):
    print("pdf'i okumaya başladım...")
    doc = fitz.open(pdf_yolu)
    tam_metin = ""
    for sayfa in doc:
        tam_metin += sayfa.get_text("text")

    parent_hash_map = {}
    child_parcalar = []
    child_to_parent_map = {}

    parent_id = 0
    parent_adim = parent_boyutu - parent_kesisim
    for i in range(0,len(tam_metin),parent_adim):
        parent= tam_metin[i : i + parent_boyutu].strip()
        if len(parent) > 30:
            # Haritaya (Hash Map) kimlikleri kaydet! 
            parent_hash_map[parent_id] = parent
            parent_id= parent_id + 1


    child_id = 0
    child_adim = child_boyutu - child_kesisim
    for p_id, p_metin in parent_hash_map.items():
        for j in range(0,len(p_metin),child_adim):
            child = p_metin[j : j + child_boyutu].strip()
            if len(child) > 30:
                # 1. Normal listeye yazıyı ekle
                child_parcalar.append(child)
                
                # 2. Haritaya (Hash Map) kimlikleri kaydet! 
                child_to_parent_map[child_id] = p_id
                
                child_id = child_id + 1



    print("pdf işi tamam " + str(len(child_parcalar)) + " tane parça çıktı")
    return child_parcalar, child_to_parent_map, parent_hash_map

pdf_dosyasi = "" #Dosyanızın konumunu buraya girin. / Enter the location of your file here.

index_dosyasi = "veritabani_yeni.index"
metin_dosyasi = "veritabani_yeni.pkl"


#E5 MODELİ Çağırıyorum



# ==========================================
# 2 ve 3. KAYIT KONTROLÜ VE FAISS
# ==========================================

#INDEX KAYIT FONKSİYONU

def DATA_SET_UPDATE(pdf_yolu):
    print("Tüm veriler update ediliyor...")
    global arama_motoru
    global child_parcalar
    global child_to_parent_map
    global parent_hash_map

    child_parcalar, child_to_parent_map, parent_hash_map = pdf_okuyucu(pdf_yolu)
    kategori_metinleri = ["passage: " + metin for metin in child_parcalar]

    print("metinleri sayılara çeviriyorum, az sürebilir...")
    pdf_vektorleri = model.encode(kategori_metinleri, normalize_embeddings=True, convert_to_numpy=True)
    print("vektör işi bitti.")
    
    print("faiss veritabanını kuruyorum...")
    vektor_boyutu = pdf_vektorleri.shape[1] 
    arama_motoru = faiss.IndexFlatIP(vektor_boyutu)
    arama_motoru.add(pdf_vektorleri)
    print( str(arama_motoru.ntotal) + " tane parçayı veritabanına attık")
    
    
    #Diske Kaydetme
    print("Gelecek sefer için veritabanı diske kaydediliyor...")
    faiss.write_index(arama_motoru, index_dosyasi)
        
    with open(metin_dosyasi, "wb") as f:
        pickle.dump({
            "child_parcalar": child_parcalar,
            "child_to_parent_map": child_to_parent_map,
            "parent_hash_map": parent_hash_map
        },f)
            
    print("Kaydetme işlemi başarılı!")

    # --- GRADİO İÇİN EKLENEN ARACI FONKSİYON ---
def gradio_pdf_guncelle(yeni_pdf):
    if yeni_pdf is not None:
        DATA_SET_UPDATE(yeni_pdf) # Mevcut fonksiyonunu çağırıyor
        return "✅ Veritabanı başarıyla güncellendi!"
    return "Dosya seçilmedi."

while(True):
  print("\nDataset i güncellemek ister misiniz\n\n0-hayır\n1-evet" )
  x= input()
  if os.path.exists(index_dosyasi) and os.path.exists(metin_dosyasi) and (x=="0"):
    print("Kaydedilmiş veritabanı bulundu! Vektör çevirisi atlanıyor...")
    
    arama_motoru = faiss.read_index(index_dosyasi)
    
    with open(metin_dosyasi, "rb") as f:
        kayitli_veri = pickle.load(f)
        child_parcalar = kayitli_veri["child_parcalar"]
        child_to_parent_map = kayitli_veri["child_to_parent_map"]
        parent_hash_map = kayitli_veri["parent_hash_map"]


    print("E5 modeli müşteri sorusu için yükleniyor...")
    model_yolu = "intfloat/multilingual-e5-base"
    model = SentenceTransformer(model_yolu, device='cuda')
    break


  elif(x=="1"):
    

    print("e5 modeli yükleniyor...")
    model_yolu = "intfloat/multilingual-e5-base"
    model = SentenceTransformer(model_yolu, device='cuda')

    DATA_SET_UPDATE(pdf_dosyasi)
    break


# ==========================================
# 4. RAG VE GPT BAĞLANTISI 
# ==========================================
client = OpenAI()

while(True):
    
   print("\n\nHangi Modeli Kullanmak İstiyorsun\n\n0-gpt-4o-mini\n1-gpt-5.6 Luna")
   y=input()
   if(y=="1" or y=="0"):
       break
   else:
       continue
      
def rag_asistanina_sor(musteri_sorusu, k_adet=6):
    print("müşteri şunu sordu:"+ str(musteri_sorusu))
    
    soru_vektoru = model.encode(["query: " + str(musteri_sorusu)], normalize_embeddings=True, convert_to_numpy=True)
    skorlar, indeksler = arama_motoru.search(soru_vektoru, k_adet)
    
    bulunan_parent_metinler = []
    gorulen_parent_idler = set()

    for child_index in indeksler[0]:
       if child_index != -1: #Geçerli index kontrolü
           p_id = child_to_parent_map[child_index]

           #Aynı parent metni yine GPT ye gitmesin
           if p_id not in gorulen_parent_idler:
               gorulen_parent_idler.add(p_id)
               bulunan_parent_metinler.append(parent_hash_map[p_id])
        
    birlestirilmis_baglam = "\n\n---\n\n".join(bulunan_parent_metinler)
    
    print("gpt'ye istek atıyorum...")
    
# Sistem Mesajı: Asistanın kimliği ve kuralları
    luna_system_prompt = """Sen uzman, kibar ve profesyonel bir asistanısın.

Cevap üretirken şu öncelik sırasına kesinlikle uy:

TEMEL GÖREV:
Kullanıcının sorularını önce verilen BAĞLAM ı
inceleyerek cevapla.

ÖNCELİK (BAĞLAM): Sana verilen BAĞLAM metninde sorunun cevabı varsa, cevabı doğrudan ve eksiksiz olarak BAĞLAM a dayandırarak ver. Kendi eğitim setini kullanma.
(Altına not düş "Veriler tamamen PDF den çekildi"(koyu renkte yaz)

 Kurallar:

1. WEB ARAMASI

Bağlamda cevap yoksa ilk olarak hemen Web araması yap.

Güncel bilgiyi internetten araştır ve mümkün olduğunca
güvenilir ve birincil kaynakları tercih et.

Özellikle:
- resmi üniversite siteleri
- resmi devlet kurumları
- YÖK
- INSA Lyon
- resmi OpenAI belgeleri
- resmi şirket/kurum sayfaları

gibi kaynakları önceliklendir.

2. PDF + WEB BİRLİKTE

Bir soru hem PDF'deki bilgilerden hem de güncel internet
bilgisinden yararlanmayı gerektiriyorsa iki kaynağı ayrı ayrı
değerlendir.
Bağlam kullanıldıysa ama Bağlam bilgi için yeterli değilse Bağlam ile birlikte mutlaka WEB araması yap:

**Veriler PDF + Web araştırması kullanılarak oluşturuldu**(Koyu renkte yaz) notunu ekle.

PDF'den gelen bilgi ile web'den gelen bilgiyi birbirine
karıştırma.

3. KAYNAK AYRIMI

Sadece Web araması yapıldıysa cevap sonunda:
**Web araması kullanıldı**(Koyu Renkte) notunu ekle.


5. GÜNCEL BİLGİLERDE TAHMİN YAPMA

Güncel veri gerekiyorsa web araştırması yapılmadan kesin
bir bilgi verme.

6. SOHBET (CHIT-CHAT):
Kullanıcı selamlaşma veya hal hatır sorma gibi sohbet amaçlı yazarsa samimi ve kısa bir karşılık ver, ardından nasıl yardımcı olabileceğini sor.


7. CEVAP KALİTESİ

Cevap:
- açık
- detaylı
- profesyonel
- mantıksal
- gereksiz tekrar içermeyen
bir biçimde hazırlanmalıdır.

Kullanıcı özellikle karşılaştırma, tarih, ücret, başvuru,
yönetmelik veya güncel durum soruyorsa mümkün olduğunca
somut bilgiler ver.

8. WEB KULLANILDIĞINDA

Web aramasından elde edilen bilgileri doğrudan cevap üretmek
için kullan; fakat doğrulanamayan iddiaları kesin gerçek olarak
sunma.
"""


    system_prompt_4o_mini = """
Sen uzman, kibar ve profesyonel bir asistansın.

TEMEL GÖREV:
Kullanıcının sorularını verilen BAĞLAM ı
inceleyerek cevapla.

BAĞLAM): Sana verilen BAĞLAM metninde sorunun cevabı varsa, cevabı doğrudan ve eksiksiz olarak BAĞLAM a dayandırarak ver.
(Altına not düş "Veriler tamamen PDF den çekildi"(kesinlikle koyu renkte yaz)

KURALLAR:

1. PDF dışında bilgi uydurma.
BAĞLAM da cevap bulunmuyorsa kesinlikle tahmin yapma. Bu tarz cevap ver: "Bu konuda bir bilgim yok. Farklı bir konuda yardımcı olmamı ister misin?"

2. Kullanıcı selamlaşma veya hal hatır sorma gibi sohbet amaçlı yazarsa samimi ve kısa bir karşılık ver, ardından nasıl yardımcı olabileceğini sor.ap ver.

3. Cevap mümkün olduğunca oldukça detaylı ve uzun olsun.
Cümleye doğal başla (Cevap verirken bağlam kelimesi yerine PDF i kullan)

"""


    # Kullanıcı Mesajı: Bağlam ve sorunun iletildiği format
    user_promt = f"""Aşağıdaki BAĞLAM bilgilerini incele ve KULLANICI SORUSU'nu yanıtla.

BAĞLAM:
{birlestirilmis_baglam}

---
KULLANICI SORUSU:
{musteri_sorusu}
"""
#Model Seçme
    print("prompt:",user_promt)

   
    
    if(y == "0"):
            print("Gpt-4o_mini kullanılıyor...")
            
            cevap = client.chat.completions.create(
                 model="gpt-4o-mini",
                messages=[
                {"role": "system", "content": system_prompt_4o_mini},
                {"role": "user", "content": user_promt}
           ],
           temperature=0.3
           )
    
            return cevap.choices[0].message.content
            

    elif(y=="1"):
            print("Gpt_5.6_LUNA kullanılıyor...")

            cevap = client.responses.create(
                 model ="gpt-5.6-luna",
            tools=[
                {
                    "type": "web_search"
                }
            ],

            input=[
                {
                    "role": "system",
                    "content": luna_system_prompt
                },
                {
                    "role": "user",
                    "content": user_promt
                }
            ]
        )
    return cevap.output_text
           



import time

# ==========================================
# GRADIO ARAYÜZÜ
# ==========================================

def gradio_cevap_uret(mesaj, history):
    cevap = rag_asistanina_sor(mesaj)
    return cevap

def gradio_pdf_guncelle(yeni_pdf):
    if yeni_pdf is None:
        yield "Dosya seçilmedi."
        return

    global arama_motoru
    global child_parcalar
    global child_to_parent_map
    global parent_hash_map

    adim_metni = "Tüm veriler update ediliyor...\n"
    yield adim_metni
    time.sleep(0.5)

    adim_metni += "pdf'i okumaya başladım...\n"
    yield adim_metni
    time.sleep(0.2)
    
    # 1. Okuma İşlemi
    child_parcalar, child_to_parent_map, parent_hash_map = pdf_okuyucu(yeni_pdf)
    
    adim_metni += "e5 modeli yükleniyor...\n"
    yield adim_metni
    time.sleep(1.0)

    adim_metni += "metinleri vektörlere çeviriyorum, az sürebilir...\n"
    yield adim_metni
    
    # 2. Vektör İşlemi
    kategori_metinleri = ["passage: " + metin for metin in child_parcalar]
    pdf_vektorleri = model.encode(kategori_metinleri, normalize_embeddings=True, convert_to_numpy=True)
    
    vektor_boyutu = pdf_vektorleri.shape[1] 
    arama_motoru = faiss.IndexFlatIP(vektor_boyutu)
    arama_motoru.add(pdf_vektorleri)
    
    adim_metni += str(arama_motoru.ntotal) + " tane parçayı veritabanına attık\n"
    yield adim_metni
    time.sleep(0.5)
    
    # 3. Kayıt İşlemi
    faiss.write_index(arama_motoru, index_dosyasi)
    with open(metin_dosyasi, "wb") as f:
        pickle.dump({
            "child_parcalar": child_parcalar,
            "child_to_parent_map": child_to_parent_map,
            "parent_hash_map": parent_hash_map
        }, f)
        
    adim_metni += "✅ Veritabanı başarıyla güncellendi!"
    yield adim_metni


with gr.Blocks(title="PDF YARDIMCISI") as arayuz:
    gr.Markdown(
        """
        <div style="text-align: center; margin-top: 30px; margin-bottom: 20px;">
            <h1 style="color: #2c3e50; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
                💼 Akıllı Asistan
            </h1>
            <p style="color: #7f8c8d; font-size: 16px;">
                Sisteme yüklenen PDF veritabanı üzerinden sorularınızı anında yanıtlar.
            </p>
        </div>
        """
    )
    
    with gr.Row():
        pdf_yukleyici = gr.File(label="PDF Değiştir (Veritabanını Günceller)", type="filepath", file_types=[".pdf"])
        durum_kutusu = gr.Textbox(label="Güncelleme Durumu", value="Hazır.", interactive=False, lines=7)
    
    pdf_yukleyici.upload(fn=gradio_pdf_guncelle, inputs=[pdf_yukleyici], outputs=[durum_kutusu])
    
    gr.ChatInterface(
        fn=gradio_cevap_uret,
    )

print("\nArayüz başlatılıyor, tarayıcı otomatik olarak açılacak...\n")
arayuz.launch(inbrowser=True, share=True)