source .venv/bin/activate

echo Open the ssh tunnel to the server :
echo ssh -N -L 8501:localhost:8501 root@tviel-gpu.nv-stg-dgxc.teleport.sh
echo Then access the app at http://localhost:8501

streamlit run webapp/trajectory_viewer.py
