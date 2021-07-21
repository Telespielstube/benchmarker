from menu import Menu

# Prevents main.py from being imported as module, initializes the menu class and calls the show_menu function.
if __name__ == '__main__':
    menu = Menu()
    menu.show_menu(1, 5)