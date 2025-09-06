"""
Scenario: Meeting Scheduler

You are building a scheduling application. You need to find the earliest common time slot for a meeting for a group of people, given their schedules and a required meeting duration.

The schedules are provided as a list of lists, where each inner list represents a person's busy intervals. Each interval is a `[start, end]` pair, representing minutes from the start of the day.

The function should return the earliest time slot `[start, end]` of the given duration that is available for *all* attendees. If no such slot exists, it should return an empty list.

For this problem, let's assume the working day is from 9:00 AM (540) to 5:00 PM (1020). The search for a slot should be within these bounds.
"""
import stat
from typing import List

from pytest import freeze_includes

class Solution:

    def is_free(self, time, schedules, duration):
        free = True
        start = time
        end = time + duration 
        for person in schedules:
            for meeting in person:
                if meeting[0] <= start <= meeting[1]-1 or meeting[0] < end <= meeting[1] or (start < meeting[0] and end > meeting[1]):
                    start = meeting[1] 
                    free = False
                    break
            if not free:
                break
        return free, start
            

    def find_earliest_slot(self, schedules: List[List[List[int]]], duration: int) -> List[int]:
        """
        Finds the earliest available meeting slot for a group of people.

        Args:
            schedules: A list of schedules. Each schedule is a list of busy intervals
                       [start, end] for a person.
            duration: The required duration of the meeting in minutes.

        Returns:
            A list `[start, end]` representing the earliest available slot,
            or an empty list if no slot is found.
        """
        start = 540
        free = True
        while start <= 1020-duration:
            print(start)
            free, new_start = self.is_free(start, schedules, duration)
            if free:
                return  [start, start+duration]
            else:
                start = new_start
            free = True
        
        return []

if __name__ == '__main__':
    solver = Solution()

    # Test Case 1: Basic case from description
    schedules1 = [
        [[540, 600], [900, 960]],  # Person 1: 9-10 AM, 3-4 PM
        [[660, 720]]              # Person 2: 11 AM - 12 PM
    ]
    duration1 = 30
    expected1 = [600, 630] # 10:00 - 10:30 AM
    output1 = solver.find_earliest_slot(schedules1, duration1)
    print(f"Test Case 1: {'Passed' if output1 == expected1 else 'Failed'}")
    print(f"  Input Schedules: {schedules1}, Duration: {duration1}")
    print(f"  Output: {output1}, Expected: {expected1}")
    print("-" * 20)

    # Test Case 2: No possible slot
    schedules2 = [
        [[540, 720]], # 9 AM - 12 PM
        [[720, 900]]  # 12 PM - 3 PM
    ]
    duration2 = 60
    expected2 = [900, 960]
    output2 = solver.find_earliest_slot(schedules2, duration2)
    print(f"Test Case 2: {'Passed' if output2 == expected2 else 'Failed'}")
    print(f"  Input Schedules: {schedules2}, Duration: {duration2}")
    print(f"  Output: {output2}, Expected: {expected2}")
    print("-" * 20)

    # Test Case 3: Overlapping busy times
    schedules3 = [
        [[600, 660]], # 10:00 - 11:00
        [[630, 690]]  # 10:30 - 11:30
    ]
    duration3 = 30
    expected3 = [540, 570] # Earliest slot is 9:00 - 9:30
    output3 = solver.find_earliest_slot(schedules3, duration3)
    print(f"Test Case 3: {'Passed' if output3 == expected3 else 'Failed'}")
    print(f"  Input Schedules: {schedules3}, Duration: {duration3}")
    print(f"  Output: {output3}, Expected: {expected3}")
    print("-" * 20)

    # Test Case 4: Back-to-back busy times
    schedules4 = [
        [[540, 600]], # 9:00 - 10:00
        [[600, 660]]  # 10:00 - 11:00
    ]
    duration4 = 60
    expected4 = [660, 720] # 11:00 - 12:00
    output4 = solver.find_earliest_slot(schedules4, duration4)
    print(f"Test Case 4: {'Passed' if output4 == expected4 else 'Failed'}")
    print(f"  Input Schedules: {schedules4}, Duration: {duration4}")
    print(f"  Output: {output4}, Expected: {expected4}")
    print("-" * 20)

    # Test Case 5: Slot available at the end of the day
    schedules5 = [
        [[540, 960]] # 9:00 AM - 4:00 PM
    ]
    duration5 = 60
    expected5 = [960, 1020] # 4:00 PM - 5:00 PM
    output5 = solver.find_earliest_slot(schedules5, duration5)
    print(f"Test Case 5: {'Passed' if output5 == expected5 else 'Failed'}")
    print(f"  Input Schedules: {schedules5}, Duration: {duration5}")
    print(f"  Output: {output5}, Expected: {expected5}")
    print("-" * 20)

    # Test Case 6: Everyone is free
    schedules6 = [[], []]
    duration6 = 120
    expected6 = [540, 660] # 9:00 AM - 11:00 AM
    output6 = solver.find_earliest_slot(schedules6, duration6)
    print(f"Test Case 6: {'Passed' if output6 == expected6 else 'Failed'}")
    print(f"  Input Schedules: {schedules6}, Duration: {duration6}")
    print(f"  Output: {output6}, Expected: {expected6}")
    print("-" * 20)

    # Test Case 7: Bug case where proposed slot envelops a busy slot
    # This test is designed to fail the user's current implementation
    schedules7 = [[[540, 595], [600, 620]]] # Busy 9:00-9:55, and 10:00-10:20
    duration7 = 30
    expected7 = [620, 650] # Earliest slot should be 10:20 - 10:50
    output7 = solver.find_earliest_slot(schedules7, duration7)
    print(f"Test Case 7: {'Passed' if output7 == expected7 else 'Failed'}")
    print(f"  Input Schedules: {schedules7}, Duration: {duration7}")
    print(f"  Output: {output7}, Expected: {expected7}")
    print("-" * 20)
