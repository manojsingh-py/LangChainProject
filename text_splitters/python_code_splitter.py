from langchain_text_splitters import RecursiveCharacterTextSplitter, Language


text = """
        Most literals are compared by equality, however the singletons True, False and None are compared by identity.
        
        Patterns may use named constants. These must be dotted names to prevent them from being interpreted as capture variables:
        
        from enum import Enum
        
        class Color(Enum):
            RED = 'red'
            GREEN = 'green'
            BLUE = 'blue'
        
        color = Color(input("Enter your choice of 'red', 'blue' or 'green': "))
        
        match color:
            case Color.RED:
                print("I see red!")
            case Color.GREEN:
                print("Grass is green")
            case Color.BLUE:
                print("I'm feeling the blues :(")
        """

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=100,
    chunk_overlap=0
)

result = splitter.split_text(text)
print(len(result))
print(result)


