# Data Structures -- T1 Examination

**Maximum Marks:** 20    **Maximum Time:** 1 hour

| CO | Course outcome |
|---|---|
| CO2 | Apply and implement various linear data structures like array, linked list, stack, and queue in different problems and applications |
| CO3 | Analyse the performance of various sorting and searching techniques |
| CO4 | Demonstrate and implement various operations like search, traverse, insertion, deletion, etc. on different non-linear data structures |

**Note:** Attempt all questions.

**Q1.** Consider an empty stack S implemented using a one‑dimensional array A[1..5] (index 1 is the bottom of the stack). Perform the following sequence of operations and, after each operation, write the contents of the array from index 1 to 5, indicating the top element with a ‘T’ and empty positions with ‘–’. 

1. PUSH(3) 
2. PUSH(7) 
3. PUSH(2) 
4. POP() 
5. PUSH(9) 
6. POP()  **[4 marks] [CO2, Apply]**


**Q2.** Given the input array A = [3, 0, 2, 5, 4, 1, 2, 3] where each element is a non‑negative integer not larger than k = 5, perform counting sort as described in the textbook (Section 8.2). Assume the algorithm uses arrays A[1..n], B[1..n] for the sorted output and C[0..k] for temporary storage.  **[5 marks] [CO3, Analyze]**

(a) What is the final sorted array B produced by counting sort?  [2]
(b) What are the contents of the auxiliary array C after line 8 of the algorithm (i.e., after the loop that counts occurrences of each key) is executed?  [2]
(c) Briefly explain why counting sort runs in O(n + k) time for this input.  [1]

**Q3.** A max‑priority queue is to be implemented using a binary max‑heap stored in an array. The initial array (1‑based indexing) contains the elements {40, 35, 30, 20, 25, 15, 10}. Assume that the array representation and the heap operations follow the definitions for a max‑heap given in the textbook.  **[5 marks] [CO4, Apply]**

(a) Show the array after inserting the key 45 using the standard heap‑INSERT operation.  [2]
(b) After the insertion, perform a HEAP‑EXTRACT‑MAX and give the resulting array.  [3]

**Q4.** Consider the sequence of integer keys: 15, 6, 23, 4, 7, 14, 30, 2, 9. Write a C program fragment that reads these keys using scanf in a loop, inserts them one by one into an initially empty AVL tree, and finally prints the height of the AVL tree using printf. You must explicitly show the control‑flow constructs (loops, conditionals) and the use of appropriate data types for the keys and the height variable. Assume the standard AVL node structure is defined as:

```c
typedef struct AVLNode {
    int key;
    struct AVLNode *left, *right;
    int height;
} AVLNode;
```

Your fragment should include the scanf loop, the insertion calls, any necessary height updates, and the final printf statement. Do not write the full AVL rotation functions; just indicate the function call for insertion (e.g., `root = avlInsert(root, key);`).  **[6 marks] [CO4, Apply]**

(a) Show the exact sequence of scanf statements, the loop structure, and the calls to the insertion function for the given keys.  [3]
(b) State the height of the AVL tree after all insertions and write the corresponding printf statement that would output this height.  [3]
