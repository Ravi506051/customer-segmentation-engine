import streamlit as st, pandas as pd, numpy as np, plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
st.title("Customer Segmentation Engine"); r=np.random.default_rng(18); n=900
d=pd.DataFrame({"annual_spend":r.gamma(2.5,600,n),"orders_per_year":r.poisson(5,n)+1,"avg_basket":r.gamma(2,45,n),"days_since_purchase":r.integers(1,365,n)})
k=st.sidebar.slider("Clusters",2,8,4); cols=d.columns.tolist(); X=StandardScaler().fit_transform(d[cols]); d["cluster"]=KMeans(n_clusters=k,n_init=15,random_state=42).fit_predict(X).astype(str)
pc=PCA(n_components=2,random_state=42).fit_transform(X); d["PC1"]=pc[:,0]; d["PC2"]=pc[:,1]
st.metric("Customers",len(d)); st.plotly_chart(px.scatter(d,x="PC1",y="PC2",color="cluster",hover_data=cols),use_container_width=True)
st.subheader("Segment profiles"); st.dataframe(d.groupby("cluster")[cols].mean().round(1),use_container_width=True)
st.download_button("Download segments",d.to_csv(index=False),"customer_segments.csv")
