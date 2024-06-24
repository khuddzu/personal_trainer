from config_handler import ConfigHandler


def main():
    config_handler = ConfigHandler("../templates/editor.ini")
    config_handler.global_vars
    global_terms = config_handler.get_all_values()
    print(global_terms)


if __name__ == "__main__":
    main()
