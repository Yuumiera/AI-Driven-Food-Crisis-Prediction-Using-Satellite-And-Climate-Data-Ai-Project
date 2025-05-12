import ee

def init_gee():
    try:
        ee.Initialize()
        print("✅ GEE zaten başlatılmış.")
    except Exception:
        ee.Authenticate(auth_mode='notebook')
        ee.Initialize()
        print("✅ GEE oturumu başlatıldı.")
