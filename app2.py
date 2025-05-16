import pandas as pd
from sklearn.cluster import KMeans
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import altair as alt
import plotly_express as px
from sklearn.metrics import silhouette_score

from collections import Counter
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import cv2
from colorthief import ColorThief
from io import BytesIO #Belleği temizlemek için kullanılır
import tempfile


def load_image(image_file): #Fotoğraf Göstermek için bu fonksiyonu yazarız.
    img = Image.open(image_file)
    return img

def get_image_pixels(filename):
    with Image.open(filename) as rgb_image:
        image_pixel = rgb_image.getpixel((30,30))
    return image_pixel

def load_image_with_cv(image_file):
    image = Image.open(image_file)
    return cv2.cvtColor(np.array(image),cv2.COLOR_RGB2HSV)


def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(*rgb)

def prep_image(raw_img,blur=False):
    modified_image = cv2.resize(raw_img,(200,200),interpolation=cv2.INTER_AREA)
    if blur:
        modified_image = cv2.GaussianBlur(modified_image,(7,7),0)
        modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1],3)
    return modified_image



@st.cache_data
def kmeans_sil(img):
    num_clusters = [i for i in range(2, 200)]
    sample_size = int(0.10 * len(img))
    img_sample = img[np.random.choice(img.shape[0], sample_size, replace=False)]
    sil_score = []
    for num in num_clusters:
        kms = KMeans(n_clusters=num, random_state=42, n_init='auto')
        kms.fit(img_sample)
        sil_score.append(silhouette_score(img_sample, kms.labels_))
    n = num_clusters[sil_score.index(max(sil_score))+2]
    return n


def histogram_color_analysis(image,bins=20):
    image =cv2.cvtColor(np.array(image),cv2.COLOR_BGR2RGB)
    img = image.reshape(-1,3)

    reduced = (img // (256//bins)) * (256//bins)

    counts = Counter(map(tuple,reduced))

    common_colors = counts.most_common(10)

    df=pd.DataFrame({
        'RGB': [color for color,count in common_colors],
        'Counts': [count for color,count in common_colors],
        'Hex': [rgb_to_hex(color) for color,count in common_colors]
    })
    return df

def get_color_thief_from_uploaded_file(uploaded_file):
    # Dosya içeriğini geçici bir dosyaya yaz
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    temp_file.seek(0)
    return ColorThief(temp_file.name)


def color_analysis(img,n=5):
    clf = KMeans(n)
    color_labels = clf.fit_predict(img) #Modeli eğitirim
    center_colors = clf.cluster_centers_ #Cluster olarak renkleri alırım.
    counts = Counter(color_labels) #Burada renklerin sayısını çıkartırım.
    ordered_colors = [center_colors[i] for i in counts.keys()] #Burada counts yapısı içerisinde değerleri alırım.
    hex_colors = [rgb_to_hex(ordered_colors[i]) for i in counts.keys()] #Burada hex kodlarını alırım ve bunları bir liste olarak tutarım
    df = pd.DataFrame({'labels': hex_colors, 'Counts': counts.values()})
    return df



def main():
    st.title('Fotoğraf Renk Paleti')
    menu = ['Anasayfa','Hakkında']
    choice = st.sidebar.selectbox("Menu",menu)
    if choice == "Anasayfa":
        st.subheader("Anasayfa")
        image_file = st.file_uploader("Upload Image", type=['PNG', 'JPG', 'JPEG'],accept_multiple_files=True)
        if image_file:
            if len(image_file) > 1:
                liste = []
                for im in image_file:
                    img = load_image(im)
                    st.image(img)
                    im.seek(0)  # Dosya okuma konumunu başa al
                    color_thief = get_color_thief_from_uploaded_file(im)
                    dominant_color = color_thief.get_color(quality=1)
                    dominant_col = rgb_to_hex(dominant_color)
                    palette = color_thief.get_palette(color_count=6)
                    hex_palette = [rgb_to_hex(color) for color in palette]

                    liste.append(pd.DataFrame(
                        {'Dosyanın_ismi': im.name, 'Dominant renk:': [dominant_col], 'Renk Paleti:': [hex_palette]}))
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.header('En çok bulunan Renkler')
                        for hex_color in hex_palette:
                            st.markdown(
                                f"""
                                                <div style='display: flex; align-items: center; margin-bottom: 5px;'>
                                                    <div style='width: 50px; height: 30px; background-color: {hex_color}; border: 1px solid #000;'></div>
                                                    <span style='margin-left: 10px; font-family: monospace;'>{hex_color}</span>
                                                </div>
                                                """,
                                unsafe_allow_html=True
                            )

                    with col2:
                        st.header('Hakim Renk')
                        st.markdown(
                            f"""
                                                        <div style='display: flex; align-items: center; margin-bottom: 10px;'>
                                                            <div style='width: 50px; height: 30px; background-color: {dominant_col}; border: 1px solid #000;'></div>
                                                            <span style='margin-left: 10px; font-family: monospace;'>{dominant_col}</span>
                                                        </div>
                                                        """,
                            unsafe_allow_html=True
                        )

                final_df = pd.concat(liste, ignore_index=True)
                st.header('Dataframe')
                st.write(final_df)
                csv = final_df.to_csv(index=False)
                st.download_button(
                    label="CSV dosyasını İndirin",
                    data=csv,
                    file_name="file.csv",
                    mime="text/csv",
                    key='download-csv'
                )
            elif len(image_file) == 1:
                # Sadece bir dosya yüklendi
                im = image_file[0]
                img = load_image(im)
                st.image(img)

                image_pixel = get_image_pixels(im)
                st.write(image_pixel)

                im.seek(0)
                color_thief = get_color_thief_from_uploaded_file(im)
                dominant_color = color_thief.get_color(quality=1)
                dominant_col = rgb_to_hex(dominant_color)
                palette = color_thief.get_palette(color_count=6)
                hex_palette = [rgb_to_hex(color) for color in palette]

                df = pd.DataFrame({
                    'Dosyanın_ismi': im.name,
                    'Dominant renk:': [dominant_col],
                    'Renk Paleti:': [hex_palette]
                })

                # Renkleri gösterme
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.header('En çok bulunan Renkler')
                    for hex_color in hex_palette:
                        st.markdown(
                            f"""
                                    <div style='display: flex; align-items: center; margin-bottom: 5px;'>
                                        <div style='width: 50px; height: 30px; background-color: {hex_color}; border: 1px solid #000;'></div>
                                        <span style='margin-left: 10px; font-family: monospace;'>{hex_color}</span>
                                    </div>
                                    """,
                            unsafe_allow_html=True
                        )

                with col2:
                    st.header('Hakim Renk')
                    st.markdown(
                        f"""
                                <div style='display: flex; align-items: center; margin-bottom: 10px;'>
                                    <div style='width: 50px; height: 30px; background-color: {dominant_col}; border: 1px solid #000;'></div>
                                    <span style='margin-left: 10px; font-family: monospace;'>{dominant_col}</span>
                                </div>
                                """,
                        unsafe_allow_html=True
                    )

                st.header('Dataframe')
                st.write(df)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="CSV Dosyasını İndirin",
                    data=csv,
                    file_name="file.csv",
                    mime="text/csv",
                    key='download-csv'
                )

            else:
                st.warning("Lütfen bir dosya yükleyin.")
    else:
        st.subheader("Hakkında")
        st.write('Bu web uygulaması, kar amacı güdülmeden geliştirilmiştir. Amacım; tez çalışmaları, sanatsal projeler ve ticari içerikler üreten kullanıcıların renk analiz süreçlerini kolaylaştırmak ve onları bu süreçte mutlu etmektir.Kullanıcı dostu arayüzü ve pratik araçlarıyla, renk uyumu ve dengesi konusunda hızlı ve güvenilir analizler sunmayı hedefliyoruz. Geliştirme sürecinde kullanıcı deneyimi ön planda tutulmuş, herkesin rahatlıkla kullanabileceği sade bir yapı benimsenmiştir.Görsel estetikle ilgilenen herkes için faydalı bir kaynak olmasını diliyoruz.İş birliği ve önerileriniz için cagandemirmr@gmail.com adresine mail gönderebilirsiniz.')


if __name__ == '__main__':
    main()