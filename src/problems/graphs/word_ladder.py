import math
from typing import List

class Solution:
    def is_one_letter_diff(self, word1, word2):
        diff_count = 0
        for c1, c2 in zip(word1, word2):
            if c1 != c2:
                diff_count += 1
                if diff_count > 1:
                    return False
        return diff_count == 1

    def ladderLengthDFS(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        # TODO: Implement the solution here

        def ladder_length_recurse(current_ladder_len, word, end_word, wordList):
            if word==end_word or current_ladder_len>len(wordList):
                return current_ladder_len
            ladder_lens = [math.inf]
            for word2 in wordList:
                
                if self.is_one_letter_diff(word, word2):
                    ladder_lens.append(ladder_length_recurse(current_ladder_len+1, 
                                                word2, 
                                                end_word, 
                                                wordList))
                
            return min(ladder_lens)
        
        ladder_len = ladder_length_recurse(1, beginWord, endWord, wordList)

        return ladder_len if ladder_len<=len(wordList) else 0 
    
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        
        ladders = [[beginWord]]
        while ladders:
            current_ladder = ladders.pop(0)
            word = current_ladder[-1]
            for word2 in wordList:
                if self.is_one_letter_diff(word, word2) and word2 not in current_ladder:
                    if word2 == endWord:
                        return  len(current_ladder) + 1

                    ladders.append(current_ladder + [word2])
        return 0 


if __name__ == "__main__":
    solution = Solution()
    
    # Test case 1
    beginWord = "hit"
    endWord = "cog"
    wordList = ["hot","dot","dog","lot","log","cog"]
    result = solution.ladderLength(beginWord, endWord, wordList)
    print(f"Test 1: {result}")  # Expected: 5
    
    # Test case 2
    beginWord = "hit"
    endWord = "cog"
    wordList = ["hot","dot","dog","lot","log"]
    result = solution.ladderLength(beginWord, endWord, wordList)
    print(f"Test 2: {result}")  # Expected: 0
    
    # Test case 3
    beginWord = "a"
    endWord = "c"
    wordList = ["a","b","c"]
    result = solution.ladderLength(beginWord, endWord, wordList)
    print(f"Test 3: {result}")  # Expected: 2
    
    # Test case 4
    beginWord = "hot"
    endWord = "dog"
    wordList = ["hot","dog"]
    result = solution.ladderLength(beginWord, endWord, wordList)
    print(f"Test 4: {result}")  # Expected: 0 (no transformation possible)
