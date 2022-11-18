import xml.etree.ElementTree as ET

def generate_XML():
    mujoco = ET.element("mujoco")
    mujoco.attrib = {"model": "MuJoCo Model"}

    tree = ET.ElementTree(mujoco)
    return tree

if __name__ == "__main__":
    tree = generate_XML
    filename = "temp_env.xml"
    with open (filename, "wb") as files :
        tree.write(files)