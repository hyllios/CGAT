for ((i=0; i < 2830000; i += 10000)) do
    python CGAT/prepare_volume_data.py --file data_"$i"_`expr $i + 10000`.pickle.gz &
done
