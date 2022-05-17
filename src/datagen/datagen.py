from datagen.synthetic.single import sphere, swissroll
from datagen.synthetic.multiple import intertwinedswissrolls, concentricspheres

dtype = {
    "single-sphere": sphere.RandomSphere,
    "single-swissroll": swissroll.RandomSwissRoll,
    "ittw-swissrolls": intertwinedswissrolls.IntertwinedSwissRolls,
    "inf-ittw-swissrolls": intertwinedswissrolls.IntertwinedSwissRolls,
    "conc-spheres": concentricspheres.ConcentricSpheres,
    "inf-conc-spheres": concentricspheres.ConcentricSpheres
}


