KG

*)
Install: OpenNRE
https://github.com/thunlp/OpenNRE

conda activate <your-python-environment>
git clone https://github.com/thunlp/OpenNRE.git
pip install -r requirements.txt
python setup.py install 


*)
Install: openjdk-11-jdk
sudo apt-get install openjdk-11-jdk


*)
Install: Neo4j
https://www.digitalocean.com/community/tutorials/how-to-install-and-configure-neo4j-on-ubuntu-20-04#:~:text=Run%20the%20following%20command%20to,%7C%20sudo%20apt%2Dkey%20add%20%2D&text=Next%2C%20add%20the%20Neo4j%204.1,.neo4j.com%20stable%204.1%22

# use newer: stable 4.2 instead of stable 4.1
sudo add-apt-repository "deb https://debian.neo4j.com stable 4.2"

