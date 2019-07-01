import csv
# Wir verwenden als Testdatensatz die Datai "babanas-1-2d,train.csv"
with open("bananas-1-2d.train.csv", "r") as csv_file:
    lines = csv_file.readlines()
v_list = []
fc_list = []
sc_list = []
d_list = []
v_list_close = []

for line in lines:
    data = line.split(',')
    v_list.append(float(data[0]))       #v_list ist eine Liste mit den Werten der ersten Spalte der Datei
    fc_list.append(float(data[1]))      # Analog enthalten fc_list ("first coordinate") und sc_list ("second
    sc_list.append(float(data[2]))      # coordinate") die Werte der zweiten und dritten Spalte der Datei.
def neighbor(x_1,x_2,k):
# Die Funktion neighbor ist unsere gewünschte Funktion mit drei Eingaben: erste und zweite Koordinate des Punktes,
# sowie der Parameter k. Zunächst wird der Abstand (distance) zwischen x und allen Punkten gemessen und
# in deine Liste d_list zusammengefasst.
    i = 0
    while i <= len(fc_list)-1:
        distance = (((x_1-fc_list[i])**2)+((x_2-sc_list[i])**2))**(1/2)
        d_list.append(distance)
        i = i+1
# Danach werden die Einträge der Liste mit dem kleinsten Wert beginnend sortiert. Die so neu entstandene
# Liste bezeichnen wir mit der Variabel "sort_d_list"
    sort_d_list = sorted(d_list)
# Wir gehen die ersten k Einträge von sort_d_list durch und vergleichen jeden einzeln mit allen Werten
# der ursprünglichen unsortierten Liste d_list. Falls zwei Werte übereinstimmen, wird der Klassifikationswert
# aus v_list in eine neue Liste V_list_close ("closest point") eingetragen
    s=0
    while s <= k-1:
        j = 0
        while j <= len(fc_list)-1:
            if sort_d_list[s] == d_list[j]:
                v_list_close.append(v_list[j])
                break
            else:
                j = j+1
        s = s+1
# Die so gesammelten Werte aus v_list_close werden anschließen addiert und dann das Signum der Summe ausgegeben.
    c = sum(v_list_close)
    if c < 0:
        return -1
    else:
        return 1