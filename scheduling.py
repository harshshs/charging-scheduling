import numpy as np
import pandas as pd
from datetime import datetime

class EVChargingSchedulerHGSO:
    def __init__(self, ev_data, time_slots, rtt, energy_prices):
        self.ev_data = ev_data
        self.num_evs = len(ev_data)
        self.num_stations = len(ev_data['cs_no'].unique())  # Number of unique charging stations
        self.time_slots = time_slots
        self.rtt = rtt
        self.energy_prices = energy_prices

    def fitness_function(self, solution):
        total_cost = 0
        for ev, station in enumerate(solution):
            total_cost += self.optimize_schedule(ev, station)
        return total_cost

    def optimize_schedule(self, ev, station):
        arrival_time = self.ev_data.iloc[ev]['arr_time_slot']
        departure_time = self.ev_data.iloc[ev]['dep_time_slot']
        initial_soc = self.ev_data.iloc[ev]['initial_soc']
        target_soc = self.ev_data.iloc[ev]['target_soc']

        soc = initial_soc
        cost = 0
        for t in range(self.time_slots):
            if arrival_time <= t < departure_time:
                if soc < target_soc:
                    soc += self.rtt[t]  # Charging (G2V)
                    cost += self.energy_prices[t] * self.rtt[t]
                elif soc > target_soc:
                    soc -= self.rtt[t]  # Discharging (V2G)
                    cost -= self.energy_prices[t] * self.rtt[t]
            else:
                cost += 0  # Idle time

        return cost

    def hgso_optimization(self):
        num_particles = 20  # Define the number of particles (solutions)
        particles = np.random.randint(0, self.num_stations, size=(num_particles, self.num_evs))

        for iteration in range(100):  # Number of iterations
            fitness = np.array([self.fitness_function(p) for p in particles])
            best_particle = particles[np.argmin(fitness)]  # Best solution

            # Dissolution process in HGSO (random movement towards better solutions)
            for i in range(num_particles):
                if np.random.rand() < 0.1:  # Probability of updating a particle
                    particles[i] = best_particle + np.random.randint(-1, 2, size=self.num_evs)
                    particles[i] = np.clip(particles[i], 0, self.num_stations - 1)  # Ensure valid stations

        return best_particle  # Best solution found

    def enforce_v2g_slots(self, ev_schedule, station):
        # Define the V2G slots for each charging station
        v2g_slots_cs1 = [self.time_slots - 3, self.time_slots - 4, self.time_slots - 5]  # 3rd last, 4th last, 5th last
        v2g_slots_cs2 = [1, 3]  # 2nd, 4th
        v2g_slots_cs3 = [self.time_slots - 2, self.time_slots - 3, 2]  # 2nd last, 3rd last, 3rd

        if station == 1:
            for t in v2g_slots_cs1:
                ev_schedule[t] = (t, "V2G")
        elif station == 2:
            for t in v2g_slots_cs2:
                ev_schedule[t] = (t, "V2G")
        elif station == 3:
            for t in v2g_slots_cs3:
                ev_schedule[t] = (t, "V2G")

    def schedule_evs(self):
        # Run HGSO to get the optimal station assignments
        optimal_assignments = self.hgso_optimization()
        schedule = []
        
        for ev in range(self.num_evs):
            station = optimal_assignments[ev]
            ev_schedule = []
            soc = self.ev_data.iloc[ev]['initial_soc']
            target_soc = self.ev_data.iloc[ev]['target_soc']
            
            for t in range(self.time_slots):
                if self.ev_data.iloc[ev]['arr_time_slot'] <= t < self.ev_data.iloc[ev]['dep_time_slot']:  # Within time window
                    if soc < target_soc:
                        ev_schedule.append((t, "G2V"))
                        soc += self.rtt[t]  # Charging
                    elif soc > target_soc:
                        ev_schedule.append((t, "V2G"))
                        soc -= self.rtt[t]  # Discharging
                    else:
                        ev_schedule.append((t, "Idle"))
                else:
                    ev_schedule.append((t, "Idle"))
            
            # Enforce predefined V2G slots based on the station
            self.enforce_v2g_slots(ev_schedule, station)

            schedule.append({"EV": self.ev_data.iloc[ev]['ev_no'], "Station": station, "Schedule": ev_schedule})

        return schedule

# Helper function to convert time strings to time slots
def time_to_slot(time_str):
    time_format = "%I:%M %p"  # Time format: e.g., "8:00 AM"
    start_time = datetime.strptime("6:00 AM", time_format)  # Earliest time slot start
    current_time = datetime.strptime(time_str, time_format)
    delta = (current_time - start_time).seconds // 1800  # Each slot is 30 minutes
    return delta

# Main function to load EV data from a CSV and perform the scheduling
def main():
    # Manually define your CSV data as a string
    csv_data = """ev_no,arr_time,dep_time,initial_soc,target_soc,cs_no
1,8:00 AM,12:00 PM,30,80,1
2,9:30 AM,1:30 PM,40,90,1
3,7:45 AM,11:45 AM,35,85,2
4,10:00 AM,2:00 PM,25,75,3
5,6:30 AM,10:30 AM,50,95,1
6,8:15 AM,12:15 PM,20,70,2
7,7:00 AM,11:00 AM,45,85,3
8,9:00 AM,1:00 PM,30,80,1
9,7:30 AM,11:30 AM,40,90,2
10,8:45 AM,12:45 PM,50,100,3
11,10:15 AM,2:15 PM,35,85,1
12,7:15 AM,11:15 AM,25,75,2
"""
    # Convert the CSV data string into a pandas DataFrame
    from io import StringIO
    ev_data = pd.read_csv(StringIO(csv_data))

    # Convert times to time slots
    ev_data['arr_time_slot'] = ev_data['arr_time'].apply(time_to_slot)
    ev_data['dep_time_slot'] = ev_data['dep_time'].apply(time_to_slot)

    # Define parameters
    time_slots = 24  # Representing 30-minute slots
    rtt = np.random.rand(time_slots)  # Random RTT values for each time slot
    energy_prices = np.random.rand(time_slots)  # Random energy prices for each time slot

    # Initialize the scheduler with the EV data
    scheduler = EVChargingSchedulerHGSO(ev_data, time_slots, rtt, energy_prices)

    # Run the scheduling
    optimal_schedule = scheduler.schedule_evs()

    # Save the optimal schedule to a CSV file
    result_df = pd.DataFrame([{
        "EV": ev_schedule["EV"],
        "Station": ev_schedule["Station"],
        "Schedule": ev_schedule["Schedule"]
    } for ev_schedule in optimal_schedule])

    result_df.to_csv("optimized_ev_schedule.csv", index=False)

    # Print results
    for ev_schedule in optimal_schedule:
        print(f"EV {ev_schedule['EV']} assigned to Charging Station {ev_schedule['Station']}")
        for entry in ev_schedule['Schedule']:
            t_slot = entry[0]
            action = entry[1]
            start_time = f"{6 + t_slot // 2:02d}:{'00' if t_slot % 2 == 0 else '30'}"
            end_time = f"{6 + (t_slot + 1) // 2:02d}:{'00' if (t_slot + 1) % 2 == 0 else '30'}"
            print(f"  {start_time} - {end_time}: {action}")

if __name__ == "__main__":
    main()
