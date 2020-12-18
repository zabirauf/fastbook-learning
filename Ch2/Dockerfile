FROM pytorch/pytorch

EXPOSE 8501

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY . .

CMD streamlit run ch2_sl_web.py
