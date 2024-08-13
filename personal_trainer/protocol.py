from config_handler import ConfigHandler


def main():
    config_handler = ConfigHandler("../templates/editor.ini")
    #config_handler.global_vars
    terms = config_handler.load_config()
    print(terms)


if __name__ == "__main__":
    main()
