import os

import geoip2.database
import socket

# Define our list of data centers with lat/long position
data_centers = [
    {'name': 'New York', 'lat': -40.7081, 'long': -74.0134, 'ip': "174.127.110.33"},
    {'name': 'Montreal', 'lat': 45.508840, 'long': -73.587810, 'ip': "68.168.115.43"},
    {'name': 'Amsterdam', 'lat': 52.378502, 'long': 4.899980, 'ip': "51.158.152.202"},
    {'name': 'Singapore', 'lat': 1.289987, 'long': 103.850281, 'ip': "18.132.219.32"},
    {'name': 'Nanjing', 'lat': 32.061670, 'long': 118.777990, 'ip': "218.92.243.74"},
]

# Set up the GeoIP2 reader
reader = geoip2.database.Reader('/Users/mac/Downloads/GeoLite2-City.mmdb')

# Look up the location of an IP address, the address here is the server address for Netvigator
ip_address = socket.gethostbyname('218.102.32.232')
response = reader.city(ip_address)


#
# Find the nearest data center based on lat/long position
def find_nearest_dc(latitude, longitude):
    distances = []
    for dc in data_centers:
        d_lat = abs(latitude - dc['lat'])
        d_long = abs(longitude - dc['long'])
        distance = (d_lat ** 2 + d_long ** 2) ** 0.5
        distances.append({'name': dc['name'], 'distance': distance})
    return min(distances, key=lambda d: d['distance'])


# Route traffic to the nearest data center based on location
nearest_dc = find_nearest_dc(response.location.latitude, response.location.longitude)
print(f"Routing traffic for {ip_address} to {nearest_dc['name']}")


# Monitor traffic routing
def ping(dc):
    response = os.system("ping -c 1 " + dc['ip'])
    if response == 0:
        print(f"{dc['ip']} {dc['name']} is reachable")
    else:
        print(f"{dc['name']} is down")


for dc in data_centers:
    ping(dc)

if __name__ == '__main__':
    print()
